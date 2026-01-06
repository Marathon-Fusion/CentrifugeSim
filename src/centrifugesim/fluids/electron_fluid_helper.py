import numpy as np
import numba
from numba import njit

from centrifugesim import constants

# Small floors to avoid division-by-zero in edge cells
NU_FLOOR = 1e-30
B_FLOOR  = 1e-5

#################################################################################
################################### Helper kernels ##############################
#################################################################################

@njit(cache=True)
def _kBT(Te, Te_is_eV):
    """Return k_B T in Joules (same shape as Te)."""
    if Te_is_eV:
        # k_B*T = Te[eV]*q_e
        return Te * constants.q_e
    else:
        return Te * constants.kb

@njit(nogil=True)
def get_anomalous_collision_frequency(ne, Te, Ti, J_mag, mi):
    """
    Compute anomalous electron collision frequency (1/s) due to ion-acoustic
    instability, using Bychenkov-Silin approximate scaling.
    Inputs are arrays Te [K], Ti [K], ne [m^-3], J_mag [A/m^2] with identical shapes.
    Ion mass mi [kg] is a scalar.
    """
    qe = constants.q_e
    me = constants.m_e

    # Calculate Drift Velocity
    v_drift = J_mag / (ne * qe)

    # Calculate Thermal/Sound Speeds
    # v_th_e = sqrt(2*Te/me)
    # c_s = sqrt(Te/mi)
    v_th_e = np.sqrt(2.0 * constants.kb * Te / me)
    c_s    = np.sqrt(constants.kb * Te / mi)

    # Calculate Plasma Frequency
    w_pe = np.sqrt(ne * qe**2 / (me * constants.ep0))

    # Temperature Ratio (The "Gain" knob)
    # Real physics: scales with Te/Ti.
    # Safety: Clamp Ti_eff to at least 0.1 eV to prevent Te/Ti -> Infinity
    Ti_eff = np.where(Ti < 0.1*11604, 0.1*11604, Ti)
    Tratio = Te / Ti_eff
    
    # Bychenkov-Silin Scaling (Approximate)
    # This roughly matches the kinetic theory limit without free parameters
    # Factor ~ 0.01 comes from Sagdeev saturation theory
    alpha_eff = 1.0e-2 * (Tratio) * (v_drift / v_th_e)
    
    nu_anom = alpha_eff * w_pe
    
    # Buneman Limit (Safety Cap)
    # If v_drift is HUGE (> v_th_e), the instability changes to Buneman.
    # The growth rate saturates at roughly (me/mi)^(1/3) * w_pe.
    # This prevents the term from going to infinity if density drops to zero.
    nu_max = 0.1 * w_pe

    # Threshold Check: Ion Acoustic Instability
    # Only active if electrons move faster than the acoustic wave
    #nu_anom = np.where(v_drift > c_s, nu_anom, 0.0)
    
    return np.minimum(nu_anom, nu_max)


@njit(cache=True)
def electron_collision_frequencies(
    Te, ne, nn,
    lnLambda=12.0,
    sigma_en_m2=2.0e-19, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
    Te_is_eV=False
):
    """
    Compute electron collision frequencies (1/s):
      - nu_en: electron-neutral momentum-transfer
      - nu_ei: electron-ion (Spitzer)
      - nu_e : total = nu_en + nu_ei
    Inputs are arrays Te [K or eV], ne [m^-3], nn [m^-3] with identical shapes.
    """
    kBT = _kBT(Te, Te_is_eV)                                   # J
    vth_e = np.sqrt(8.0 * kBT / (np.pi * constants.m_e))       # m/s

    # Electron-neutral (hard-sphere-like, momentum-transfer)
    nu_en = nn * sigma_en_m2 * vth_e                           # 1/s

    # Electron-ion (Spitzer), Z=1, ni=ne
    c_num = 4.0 * np.sqrt(2.0 * np.pi) * (constants.q_e**4) * lnLambda
    c_den = 3.0 * (4.0 * np.pi * constants.ep0)**2 * np.sqrt(constants.m_e)
    nu_ei = c_num * ne / (c_den * (kBT**1.5))            # 1/s

    # Total + floors
    nu_e  = nu_en + nu_ei
    nu_en = np.maximum(nu_en, NU_FLOOR)
    nu_ei = np.maximum(nu_ei, NU_FLOOR)
    nu_e  = np.maximum(nu_e,  NU_FLOOR)
    return nu_en, nu_ei, nu_e


@njit(cache=True)
def electron_conductivities(
    Te, ne, Bmag, nu_e, nu_e_anom
):
    """
    Electron conductivity tensor components (SI, S/m):
      - sigma_par_e : parallel to B
      - sigma_P_e   : Pedersen (perpendicular, in-plane with E_perp)
      - sigma_H_e   : Hall (perpendicular, out-of-phase; negative for electrons)
    Inputs:
      Te [K or eV], ne [m^-3], nn [m^-3], Br [T], Bz [T] (same shape)
    Assumes: Z=1, ni = ne (quasineutral).
    """
    # |B|
    Bmag = np.where(Bmag < B_FLOOR, B_FLOOR, Bmag)

    # Electron gyrofrequency magnitude
    Omega_e = constants.q_e * Bmag / constants.m_e  # rad/s

    # Prefactor ne e^2 / m_e
    pref = ne * (constants.q_e * constants.q_e) / constants.m_e  # S·s/m

    # Components (electrons only)
    sigma_par_e = pref / (nu_e-nu_e_anom)  # subtract anomalous collision freq
    denom = (nu_e*nu_e + Omega_e*Omega_e)
    sigma_P_e = pref * (nu_e / denom)
    sigma_H_e = - pref * (Omega_e / denom)

    return sigma_par_e, sigma_P_e, sigma_H_e, Omega_e/nu_e  # β_e


@numba.jit(nopython=True, parallel=True, nogil=True)
def solve_step(Te, Te_new, dr, dz, r_vec, n_e, Q_Joule,
               br, bz, kappa_parallel, kappa_perp,
               Jer, Jez,
               mask, dt, ion_mass):
    """
    Advances electron temperature with specific boundary conditions:
    1. Bohm sheath loss ONLY at z=0 (j=0) and r=R_max (i=NR-1).
    2. Internal masked objects have NO Bohm loss, but allow Advection.
    3. Advection at internal masks driven by u_e (-J_e).
    4. Top boundary (z=zmax) is insulating (no flux).
    """
    NR, NZ = Te.shape
    
    # Constants
    kb = constants.kb
    qe = constants.q_e
    
    # Sheath Transmission Factor
    delta_sheath = 6.0 
    alpha = 2.5 * kb / qe

    # ITERATE OVER ALL NODES
    for i in numba.prange(0, NR):
        for j in range(0, NZ):

            # Skip masked cells or vacuum
            if not (mask[i, j] == 1 and n_e[i, j] > 0.0):
                continue

            # Pre-calculate sound speed for this node (used for BCs)
            cs_local = np.sqrt((kb * Te[i, j]) / ion_mass)
            # Only used at z=0 and r=R_max
            sheath_flux_mag = delta_sheath * n_e[i, j] * cs_local * (kb * Te[i, j])

            # =========================
            # 1. Conduction & BC Fluxes
            # =========================

            # --- Right face (i+1/2) ---
            qr_rh = 0.0
            
            # CASE A: Internal or Symmetry Interface
            if i < NR - 1:
                if mask[i+1, j] == 1:
                    # EXISTING: Full Anisotropic Stencil checks
                    if (j < NZ-1 and j > 0 and 
                        mask[i, j+1] == 1 and mask[i, j-1] == 1 and
                        mask[i+1, j+1] == 1 and mask[i+1, j-1] == 1):

                        br_rh = 0.5 * (br[i, j] + br[i+1, j])
                        bz_rh = 0.5 * (bz[i, j] + bz[i+1, j])
                        k_par_rh = 0.5 * (kappa_parallel[i, j] + kappa_parallel[i+1, j])
                        k_perp_rh = 0.5 * (kappa_perp[i, j] + kappa_perp[i+1, j])
                        k_a_rh = k_par_rh - k_perp_rh
                        k_rr_rh = k_perp_rh + k_a_rh * br_rh * br_rh
                        k_rz_rh = k_a_rh * br_rh * bz_rh
                        dT_dr_rh = (Te[i+1, j] - Te[i, j]) / dr
                        dT_dz_rh = (Te[i, j+1] - Te[i, j-1] + Te[i+1, j+1] - Te[i+1, j-1]) / (4.0 * dz)
                        qr_rh = -(k_rr_rh * dT_dr_rh + k_rz_rh * dT_dz_rh)
                    else:
                        # Simple isotropic fallback for messy plasma-plasma edges
                        k_eff = 0.5 * (kappa_perp[i, j] + kappa_perp[i+1, j])
                        qr_rh = -k_eff * (Te[i+1, j] - Te[i, j]) / dr
                
                else: 
                    # Neighbor is Masked (Solid). 
                    # User Req: Bohm ONLY at rmax/zmin. Internal mask -> No Bohm.
                    # Set conduction to 0.0 (Insulating/Advection dominated).
                    qr_rh = 0.0
            
            # CASE B: Outer Physical Wall (r = rmax)
            else: 
                # User Req: Only bohm through sheath here.
                qr_rh = sheath_flux_mag

            # --- Left face (i-1/2) ---
            qr_lh = 0.0
            
            # CASE A: Internal Interface
            if i > 0:
                if mask[i-1, j] == 1:
                    if (j < NZ-1 and j > 0 and
                        mask[i, j+1] == 1 and mask[i, j-1] == 1 and
                        mask[i-1, j+1] == 1 and mask[i-1, j-1] == 1):
                        
                        br_lh = 0.5 * (br[i, j] + br[i-1, j])
                        bz_lh = 0.5 * (bz[i, j] + bz[i-1, j])
                        k_par_lh = 0.5 * (kappa_parallel[i, j] + kappa_parallel[i-1, j])
                        k_perp_lh = 0.5 * (kappa_perp[i, j] + kappa_perp[i-1, j])
                        k_a_lh = k_par_lh - k_perp_lh
                        k_rr_lh = k_perp_lh + k_a_lh * br_lh * br_lh
                        k_rz_lh = k_a_lh * br_lh * bz_lh
                        dT_dr_lh = (Te[i, j] - Te[i-1, j]) / dr
                        dT_dz_lh = (Te[i, j+1] - Te[i, j-1] + Te[i-1, j+1] - Te[i-1, j-1]) / (4.0 * dz)
                        qr_lh = -(k_rr_lh * dT_dr_lh + k_rz_lh * dT_dz_lh)
                    else:
                        k_eff = 0.5 * (kappa_perp[i, j] + kappa_perp[i-1, j])
                        qr_lh = -k_eff * (Te[i, j] - Te[i-1, j]) / dr
                else:
                    # Neighbor is Masked -> No Bohm.
                    qr_lh = 0.0 
            
            # CASE B: Axis of Symmetry (r = 0)
            else:
                qr_lh = 0.0

            # --- Top face (j+1/2) ---
            qz_th = 0.0
            
            # CASE A: Internal Interface
            if j < NZ - 1:
                if mask[i, j+1] == 1:
                    if (i < NR-1 and i > 0 and
                        mask[i+1, j] == 1 and mask[i-1, j] == 1 and
                        mask[i+1, j+1] == 1 and mask[i-1, j+1] == 1):

                        br_th = 0.5 * (br[i, j] + br[i, j+1])
                        bz_th = 0.5 * (bz[i, j] + bz[i, j+1])
                        k_par_th = 0.5 * (kappa_parallel[i, j] + kappa_parallel[i, j+1])
                        k_perp_th = 0.5 * (kappa_perp[i, j] + kappa_perp[i, j+1])
                        k_a_th = k_par_th - k_perp_th
                        k_zz_th = k_perp_th + k_a_th * bz_th * bz_th
                        k_rz_th = k_a_th * br_th * bz_th
                        dT_dz_th = (Te[i, j+1] - Te[i, j]) / dz
                        dT_dr_th = (Te[i+1, j] - Te[i-1, j] + Te[i+1, j+1] - Te[i-1, j+1]) / (4.0 * dr)
                        qz_th = -(k_rz_th * dT_dr_th + k_zz_th * dT_dz_th)
                    else:
                        k_eff = 0.5 * (kappa_perp[i, j] + kappa_perp[i, j+1])
                        qz_th = -k_eff * (Te[i, j+1] - Te[i, j]) / dz
                else:
                    # Neighbor is Masked -> No Bohm.
                    qz_th = 0.0
            
            # CASE B: Top Boundary (z = zmax)
            else:
                # User Req: "at z=zmax no energy flux"
                qz_th = 0.0

            # --- Bottom face (j-1/2) ---
            qz_bh = 0.0
            
            # CASE A: Internal Interface
            if j > 0:
                if mask[i, j-1] == 1:
                    if (i < NR-1 and i > 0 and
                        mask[i+1, j] == 1 and mask[i-1, j] == 1 and
                        mask[i+1, j-1] == 1 and mask[i-1, j-1] == 1):

                        br_bh = 0.5 * (br[i, j] + br[i, j-1])
                        bz_bh = 0.5 * (bz[i, j] + bz[i, j-1])
                        k_par_bh = 0.5 * (kappa_parallel[i, j] + kappa_parallel[i, j-1])
                        k_perp_bh = 0.5 * (kappa_perp[i, j] + kappa_perp[i, j-1])
                        k_a_bh = k_par_bh - k_perp_bh
                        k_zz_bh = k_perp_bh + k_a_bh * bz_bh * bz_bh
                        k_rz_bh = k_a_bh * br_bh * bz_bh
                        dT_dz_bh = (Te[i, j] - Te[i, j-1]) / dz
                        dT_dr_bh = (Te[i+1, j] - Te[i-1, j] + Te[i+1, j+1] - Te[i-1, j+1]) / (4.0 * dr)
                        qz_bh = -(k_rz_bh * dT_dr_bh + k_zz_bh * dT_dz_bh)
                    else:
                        k_eff = 0.5 * (kappa_perp[i, j] + kappa_perp[i, j-1])
                        qz_bh = -k_eff * (Te[i, j] - Te[i, j-1]) / dz
                else:
                    # Neighbor is Masked -> No Bohm.
                    qz_bh = 0.0

            # CASE B: Bottom Physical Wall (z = zmin)
            else:
                # User Req: "at z=zmin... bohm"
                qz_bh = -sheath_flux_mag


            # --- Divergence (Conduction) ---
            r_center = r_vec[i] + 1e-12
            r_rh_face = r_vec[i] + 0.5 * dr
            r_lh_face = r_vec[i] - 0.5 * dr
            
            term_r = 0.0
            if i == 0:
                term_r = (r_rh_face * qr_rh) / (r_center * dr)
            else:
                term_r = (r_rh_face * qr_rh - r_lh_face * qr_lh) / (r_center * dr)
                
            div_q = term_r + (qz_th - qz_bh) / dz


            # =========================
            # 2. Advection
            # =========================
            
            # NOTE:
            # - At zmin and rmax: Advection = 0 (Bohm handled separately).
            # - At internal masked faces: Advection != 0 (Enabled).
            # - We use Te[i,j] (self) as T_up when the neighbor is a mask.

            # Right face (i+1/2)
            F_r_rh = 0.0
            if i < NR - 1:
                if mask[i+1, j] == 1:
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i+1, j])
                    T_up = Te[i+1, j] if Jr_face > 0.0 else Te[i, j]
                    F_r_rh = -alpha * T_up * Jr_face
                else:
                    # Internal Wall -> ENABLE Advection
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i+1, j])
                    T_up = Te[i, j] # Use local T for masked interaction
                    F_r_rh = -alpha * T_up * Jr_face
            else:
                # External Wall (rmax) -> User Req: No Advection (Bohm only)
                F_r_rh = 0.0

            # Left face (i-1/2)
            F_r_lh = 0.0
            if i > 0:
                if mask[i-1, j] == 1:
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i-1, j])
                    T_up = Te[i, j] if Jr_face > 0.0 else Te[i-1, j]
                    F_r_lh = -alpha * T_up * Jr_face
                else:
                    # Internal Wall -> ENABLE Advection
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i-1, j])
                    T_up = Te[i, j]
                    F_r_lh = -alpha * T_up * Jr_face
            else:
                # Axis
                F_r_lh = 0.0

            # Top face (j+1/2)
            F_z_th = 0.0
            if j < NZ - 1:
                if mask[i, j+1] == 1:
                    Jz_face = 0.5 * (Jez[i, j] + Jez[i, j+1])
                    T_up = Te[i, j+1] if Jz_face > 0.0 else Te[i, j]
                    F_z_th = -alpha * T_up * Jz_face
                else:
                    # Internal Wall -> ENABLE Advection
                    Jz_face = 0.5 * (Jez[i, j] + Jez[i, j+1])
                    T_up = Te[i, j]
                    F_z_th = -alpha * T_up * Jz_face
            else:
                # Symmetry / Top Wall (zmax) -> User Req: No energy flux
                F_z_th = 0.0

            # Bottom face (j-1/2)
            F_z_bh = 0.0
            if j > 0:
                if mask[i, j-1] == 1:
                    Jz_face = 0.5 * (Jez[i, j] + Jez[i, j-1])
                    T_up = Te[i, j] if Jz_face > 0.0 else Te[i, j-1]
                    F_z_bh = -alpha * T_up * Jz_face
                else:
                    # Internal Wall -> ENABLE Advection
                    Jz_face = 0.5 * (Jez[i, j] + Jez[i, j-1])
                    T_up = Te[i, j]
                    F_z_bh = -alpha * T_up * Jz_face
            else:
                # External Wall (zmin) -> User Req: No Advection (Bohm only)
                F_z_bh = 0.0

            # Advection Divergence
            term_adv_r = 0.0
            if i == 0:
                term_adv_r = (r_rh_face * F_r_rh) / (r_center * dr)
            else:
                term_adv_r = (r_rh_face * F_r_rh - r_lh_face * F_r_lh) / (r_center * dr)
                
            div_Fadv = term_adv_r + (F_z_th - F_z_bh) / dz
            
            # =========================
            # 3. Update
            # =========================
            rhs = -div_q - div_Fadv + Q_Joule[i, j]
            dTe_dt = (2.0 / (3.0 * n_e[i, j] * kb)) * rhs
            Te_new[i, j] = Te[i, j] + dt * dTe_dt


@numba.jit(nopython=True)
def apply_townsend_ionization_sheath(p_grid, 
                                     E_mag_grid, 
                                     Je_mag_grid, 
                                     mask, 
                                     dt,
                                     A_townsend=2.25,
                                     B_townsend=94.0):
    """
    Calculates the ionization density increment in high-field sheath regions 
    using the Townsend alpha coefficient.
    
    Physics:
    In the anode depletion zone, thermal ionization is suppressed.
    This routine calculates the field-driven ionization source term based on
    the electron beam flux (current density) and local electric field.

    # Townsend Coefficients for Atomic Hydrogen (H) as default
    
    Parameters
    ----------
    p_grid : 2D array
        Neutral gas pressure field [Pascals].
    E_mag_grid : 2D array
        Electric field magnitude [V/m].
    Je_mag_grid : 2D array
        Electron current density magnitude [A/m^2].
    mask : 2D array (int/bool)
        Geometry mask (1=Plasma, 0=Solid). Ionization skipped in Solid.
    dt : float
        Time step [s].
        
    Returns
    -------
    dni_Townsend : 2D array
        Ion density increment [m^-3] generated in this timestep.
        (Caller is responsible for creating particles from this density).
    """
    Nr, Nz = p_grid.shape
    dni_Townsend = np.zeros((Nr, Nz), dtype=np.float64)
    
    # Thresholds to avoid calculating negligible terms
    E_thresh = 500.0  # V/m
    J_thresh = 1.0    # A/m^2
    P_thresh = 0.01   # Pa
    
    qe = constants.q_e
    
    for i in range(Nr):
        for j in range(Nz):
            
            # --- 0. Geometry Check ---
            # Skip if inside solid wall
            if mask[i, j] == 0:
                continue
                
            # --- 1. Field Checks ---
            # Electric Field Magnitude
            E_mag = E_mag_grid[i, j]
            if E_mag < E_thresh: 
                continue
            
            # Electron Current Flux Magnitude
            J_mag = Je_mag_grid[i, j]
            if J_mag < J_thresh: 
                continue
            
            # Pressure Check
            p_pa = p_grid[i, j]
            if p_pa < P_thresh: 
                continue
            
            # --- 2. Townsend Alpha Calculation ---
            # alpha = A * p * exp( - B * p / E )
            
            # Reduced field argument (B * p / E)
            argument = (B_townsend * p_pa) / E_mag
            
            # If argument is too large (E is weak relative to P), probability is ~0
            if argument > 20.0: 
                continue
            
            alpha = A_townsend * p_pa * np.exp(-argument)
            
            # --- 3. Source Term ---
            # S_vol = alpha * Flux_electrons
            # Flux_electrons = |Je| / e
            # S_vol units: [1/m] * [A/m^2] / [C] = [1/m^3 s]
            S_vol = alpha * (J_mag / qe)
            
            # --- 4. Density Increment ---
            # dni = S * dt (m^-3)
            dni_Townsend[i, j] = S_vol * dt

    return dni_Townsend


# TO DO:
# use electron fluid ne_floor instead
@njit(cache=True)
def time_advance_ne_analytic_kernel_anisotropic(ne_out, ne_old, 
                                                nu_iz, nu_loss, nu_RR, beta_rec, 
                                                dt, mask):
    """
    Solves the logistic growth/decay equation analytically for density:
        dn/dt = (nu_iz - nu_loss - nu_RR) * n - beta_rec * n^2
    
    This kernel is 'unconditionally stable' (it will not oscillate or explode 
    even if dt is very large). It automatically detects if dt is large enough 
    to jump straight to equilibrium.

    Parameters
    ----------
    ne_out   : Output array (Nr, Nz)
    ne_old   : Input array (Nr, Nz) - Previous density
    nu_iz    : Input array (Nr, Nz) - Ionization frequency [1/s]
    nu_loss  : Input array (Nr, Nz) - Diffusive loss rate [1/s] (calculated externally)
    nu_RR    : Input array (Nr, Nz) - Radiative recombination frequency [1/s]
    beta_rec : Input array (Nr, Nz) - Recombination coeff [m^3/s]
    dt       : float - Time step [s]
    mask     : Input array (Nr, Nz) - 1=Plasma, 0=Solid
    """
    Nr, Nz = ne_out.shape
    
    # Exponent cap to prevent float64 overflow (exp(709) is max)
    # We use 100 as a safe "infinity" threshold. exp(100) is ~2e43.
    EXP_LIMIT = 100.0 
    
    # Minimum density floor to prevent div-by-zero or negative densities
    NE_FLOOR = 1e10 
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n0 = max(ne_old[i, j], NE_FLOOR)
                
                # 1. Net Linear Growth Rate (alpha)
                # alpha = Production - Loss
                alpha = nu_iz[i, j] - nu_loss[i, j] - nu_RR[i, j]
                
                beta = beta_rec[i, j]
                
                # --- CASE A: DECAY (alpha < 0) ---
                # The plasma is cooling/dying. 
                # Equation behaves like: n(t) -> 0
                if alpha < -1e-20:
                    # Check for underflow (if decay is too fast)
                    # arg is negative
                    arg = alpha * dt
                    
                    if arg < -EXP_LIMIT:
                        # Decayed effectively to zero (floor)
                        ne_out[i, j] = NE_FLOOR
                    else:
                        # Full Logistic Decay Solution
                        # (Formula is same as growth, but alpha is negative)
                        exp_factor = np.exp(arg) # < 1.0
                        
                        if beta > 1e-30:
                            numerator = n0 * exp_factor
                            # Denom can't be zero because alpha is negative and beta > 0
                            denominator = 1.0 + (n0 * beta / alpha) * (exp_factor - 1.0)
                            
                            # Safety check for denominator
                            if abs(denominator) > 1e-15:
                                ne_out[i, j] = numerator / denominator
                            else:
                                ne_out[i, j] = NE_FLOOR
                        else:
                            # Pure exponential decay
                            ne_out[i, j] = n0 * exp_factor

                # --- CASE B: GROWTH (alpha > 0) ---
                # The plasma is ionizing.
                # Equation behaves like: n(t) -> alpha / beta
                else:
                    arg = alpha * dt
                    
                    # 1. Check for Equilibrium (Large dt or fast rate)
                    if arg > EXP_LIMIT:
                        # We assume t -> infinity.
                        # Equilibrium = alpha / beta
                        if beta > 1e-30:
                            ne_out[i, j] = alpha / beta
                        else:
                            # Beta is 0 (No recombination) -> Runaway growth!
                            # Cap it to avoid infinity (e.g., 100x current value or some physics limit)
                            ne_out[i, j] = n0 * 100.0 
                    
                    # 2. Standard Time Advance
                    else:
                        exp_factor = np.exp(arg) # > 1.0
                        
                        if beta > 1e-30:
                             numerator = n0 * exp_factor
                             denominator = 1.0 + (n0 * beta / alpha) * (exp_factor - 1.0)
                             ne_out[i, j] = numerator / denominator
                        else:
                             # Pure exponential growth (no recombination)
                             ne_out[i, j] = n0 * exp_factor
                             
                # Final safety clamp
                if ne_out[i, j] < NE_FLOOR: 
                    ne_out[i, j] = NE_FLOOR
                
            else:
                # Masked Region (Solid)
                ne_out[i, j] = NE_FLOOR


@njit(cache=True)
def compute_ambipolar_loss_rate_anisotropic(Te, Ti, nu_in, beta_e, beta_i, 
                                            mi, kb, 
                                            inv_Lambda_z_sq, inv_Lambda_r_sq):
    """
    Computes the effective diffusive loss rate [1/s] accounting for magnetization.
    
    nu_diff = (Da_par * inv_Lambda_z^2) + (Da_perp * inv_Lambda_r^2)
    
    where:
      Da_par  = kb*(Te+Ti) / (mi*nu_in)
      Da_perp = Da_par / (1 + beta_e * beta_i)
    """
    Nr, Nz = Te.shape
    nu_loss = np.zeros_like(Te)
    
    for i in range(Nr):
        for j in range(Nz):
            # 1. Parallel Ambipolar Diffusion (Classical)
            # nu_in floor to avoid infinity
            nu_safe = max(nu_in[i, j], 1e-5)
            
            Da_par = (kb * (Te[i, j] + Ti[i, j])) / (mi * nu_safe)
            
            # 2. Perpendicular Ambipolar Diffusion (Magnetized)
            # Simon Short-Circuit / Classical Ambipolar scaling
            magnetization_factor = 1.0 + beta_e[i, j] * beta_i[i, j]
            Da_perp = Da_par / magnetization_factor
            
            # 3. Total Geometric Loss Rate
            # Loss = D_par/Lz^2 + D_perp/Lr^2
            nu_loss[i, j] = (Da_par * inv_Lambda_z_sq) + (Da_perp * inv_Lambda_r_sq)
            
    return nu_loss


@njit(cache=True)
def update_Te_local_physics(Te, ne, nn, T_n, T_i, 
                            nu_en, nu_ei, nu_iz, 
                            Q_Joule, epsilon_iz_J, dt, mi, mn, mask, Te_floor):
    """
    Updates Te based on Local Sources/Sinks (Heating + Collisions).
    Uses Semi-Implicit formulation for unconditional stability.
    """
    Nr, Nz = Te.shape
    kb = constants.kb
    me = constants.m_e
    
    # Energy Cost per ionization event (Ionization Potential + Radiation estimate)
    # epsilon_iz_J is typically ~ 13.6 eV * q_e (plus losses)
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                
                # 1. Coefficients
                n_e_local = max(ne[i, j], 1e10)
                T_old = Te[i, j]
                
                # Heat Capacity per unit volume (3/2 n k)
                Cv = 1.5 * n_e_local * kb
                
                # 2. Source Terms (Explicit)
                # Joule Heating is usually treated as constant over the step
                S_heating = Q_Joule[i, j] 
                
                # Inelastic Cooling (Ionization Energy Loss)
                # Loss = Rate * Energy_Cost
                S_inelastic = -1.0 * n_e_local * nu_iz[i, j] * epsilon_iz_J
                
                # 3. Elastic Collisions (Semi-Implicit)
                # We write: dT/dt = -nu_energy * (T - T_target)
                # T_new = (T_old + dt*nu*T_target) / (1 + dt*nu)
                
                # Frequency of energy exchange (not just momentum exchange)
                # nu_E ~ 2 * (me/M) * nu_momentum
                freq_en = 2.0 * (me / mn) * nu_en[i, j] # Assuming H atom mass
                freq_ei = 2.0 * (me / mi) * nu_ei[i, j]
                
                # Total Energy Relaxation Rate
                nu_relax = freq_en + freq_ei
                
                # Weighted Target Temperature
                if nu_relax > 1e-20:
                    T_target = (freq_en * T_n[i, j] + freq_ei * T_i[i, j]) / nu_relax
                else:
                    T_target = T_n[i, j]
                
                # 4. The Implicit Update Equation
                # Cv * (T_new - T_old)/dt = S_heat + S_inel + Cv * nu_relax * (T_target - T_new)
                # Rearranging for T_new:
                
                numerator = Cv * T_old + dt * (S_heating + S_inelastic + Cv * nu_relax * T_target)
                denominator = Cv * (1.0 + dt * nu_relax)
                
                T_new = numerator / denominator
                
                # Safety Floor
                if T_new < Te_floor:
                    T_new = Te_floor
                    
                Te[i, j] = T_new


@njit(cache=True)
def solve_Te_diffusion_implicit_SOR(Te, ne, kappa_par, kappa_perp, 
                                    br, bz, mask, dr, dz, r_coords, dt, 
                                    mi, Te_floor, max_iter=2000, tol=1e-4, omega=0.9):
    """
    Implicit Heat Diffusion Solver (SOR).
    Updates:
    - Bohm Sheath cooling applied ONLY at Physical Walls (j=0 and i=Nr-1).
    - Internal Mask boundaries are Insulating (Neumann=0).
    """
    Nr, Nz = Te.shape
    kb = constants.kb
    
    # Pre-calc geometry
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)
    
    # Sheath Parameters for BCs
    delta_sheath = 1.0 
    
    for k in range(max_iter):
        max_diff = 0.0
        
        for i in range(0, Nr):
            r_loc = r_coords[i]
            inv_r = 1.0 / (r_loc + 1e-12)
            
            for j in range(0, Nz):
                
                # Skip Mask or Empty Plasma
                if mask[i, j] == 0:
                    continue
                if ne[i, j] < 1e10:
                    continue
                
                # --- 1. Identify Neighbors & Boundary Flags ---
                # Check neighbors to see if we are connected to valid plasma
                has_N = (j < Nz-1) and (mask[i, j+1] == 1)
                has_S = (j > 0)    and (mask[i, j-1] == 1)
                has_E = (i < Nr-1) and (mask[i+1, j] == 1)
                has_W = (i > 0)    and (mask[i-1, j] == 1)
                
                # --- 2. Calculate Stencil Coefficients (Standard Diffusion) ---
                k_p = kappa_par[i, j]
                k_v = kappa_perp[i, j]
                Br = br[i, j]
                Bz = bz[i, j]
                
                k_rr = k_v + (k_p - k_v) * Br*Br
                k_zz = k_v + (k_p - k_v) * Bz*Bz
                
                # Heat Capacity Term (Inertia)
                Cv_term = 1.5 * ne[i, j] * kb / dt
                
                # Base Coefficients
                a_E = k_rr * inv_dr2 
                a_W = k_rr * inv_dr2 
                a_N = k_zz * inv_dz2 
                a_S = k_zz * inv_dz2 
                
                # Cylindrical Correction
                geom_fac = 0.5 * dr * inv_r
                a_E *= (1.0 + geom_fac)
                a_W *= (1.0 - geom_fac)
                
                # --- 3. Handle Boundaries ---
                # We strictly distinguish between PHYSICAL LIMITS and INTERNAL MASKS
                
                # NORTH (j+1)
                if not has_N: 
                    a_N = 0.0 # Insulating (Z_max or Internal Mask)
                
                # SOUTH (j-1)
                if not has_S:
                    a_S = 0.0 
                    # Only apply Bohm Sheath if we are at the PHYSICAL DOMAIN LIMIT (Z_min)
                    if j == 0:
                        cs = np.sqrt(kb * Te[i, j] / mi)
                        G_sheath = delta_sheath * ne[i, j] * cs * kb
                        Cv_term += G_sheath / dz
                    # Else: Internal mask -> Insulating (do nothing)
                
                # EAST (i+1)
                if not has_E:
                    a_E = 0.0
                    # Only apply Bohm Sheath if we are at the PHYSICAL DOMAIN LIMIT (R_max)
                    if i == Nr - 1:
                        cs = np.sqrt(kb * Te[i, j] / mi)
                        G_sheath = delta_sheath * ne[i, j] * cs * kb
                        Cv_term += G_sheath / dr
                    # Else: Internal mask -> Insulating
                    
                # WEST (i-1)
                if not has_W:
                    a_W = 0.0 # Insulating (Axis or Internal Mask)
                    
                # --- 4. SOR Update (Standard) ---
                Sigma_a = a_E + a_W + a_N + a_S
                
                T_E = Te[i+1, j] if has_E else 0.0
                T_W = Te[i-1, j] if has_W else 0.0
                T_N = Te[i, j+1] if has_N else 0.0
                T_S = Te[i, j-1] if has_S else 0.0
                
                Sum_Flux = a_E*T_E + a_W*T_W + a_N*T_N + a_S*T_S
                
                Coeff_Center = Cv_term + Sigma_a
                RHS = (1.5 * ne[i, j] * kb / dt) * Te[i, j] + Sum_Flux
                
                T_star = RHS / Coeff_Center
                
                diff = abs(T_star - Te[i, j])
                Te[i, j] = (1.0 - omega) * Te[i, j] + omega * T_star

                Te[i, j] = max(Te[i, j], Te_floor)
                
                if diff > max_diff:
                    max_diff = diff
                    
        if max_diff < tol:
            break