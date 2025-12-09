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
    Advances electron temperature with PHYSICAL boundaries at external walls
    AND internal masked objects (sheath loss).
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
            # cs = sqrt(k Te / Mi)
            cs_local = np.sqrt((kb * Te[i, j]) / ion_mass)
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
                    # NEW: Neighbor is Masked (Solid). 
                    # Flux is POSITIVE (leaves i towards i+1)
                    qr_rh = sheath_flux_mag
            
            # CASE B: Outer Physical Wall (r = rmax)
            else: 
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
                    # NEW: Neighbor is Masked (Solid).
                    # Flux is NEGATIVE (leaves i towards i-1)
                    qr_lh = -sheath_flux_mag
            
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
                    # NEW: Neighbor is Masked (Solid).
                    # Flux is POSITIVE (leaves j towards j+1)
                    qz_th = sheath_flux_mag
            
            # CASE B: Top Symmetry Plane (z = zmax)
            else:
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
                        dT_dr_bh = (Te[i+1, j] - Te[i-1, j] + Te[i+1, j-1] - Te[i-1, j-1]) / (4.0 * dr)
                        qz_bh = -(k_rz_bh * dT_dr_bh + k_zz_bh * dT_dz_bh)
                    else:
                        k_eff = 0.5 * (kappa_perp[i, j] + kappa_perp[i, j-1])
                        qz_bh = -k_eff * (Te[i, j] - Te[i, j-1]) / dz
                else:
                    # NEW: Neighbor is Masked (Solid).
                    # Flux is NEGATIVE (leaves j towards j-1)
                    qz_bh = -sheath_flux_mag

            # CASE B: Bottom Physical Wall (z = zmin)
            else:
                qz_bh = -sheath_flux_mag


            # --- Divergence ---
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
            
            # NOTE: At all solid boundaries (internal or external), we set 
            # advection flux to 0 because the enthalpy loss is accounted 
            # for in the `delta_sheath` term in the Conduction section.

            # Right face (i+1/2)
            F_r_rh = 0.0
            if i < NR - 1:
                if mask[i+1, j] == 1:
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i+1, j])
                    T_up = Te[i+1, j] if Jr_face > 0.0 else Te[i, j]
                    F_r_rh = -alpha * T_up * Jr_face
                else:
                    # Internal Wall -> Zero advection
                    F_r_rh = 0.0
            else:
                # External Wall -> Zero advection
                F_r_rh = 0.0

            # Left face (i-1/2)
            F_r_lh = 0.0
            if i > 0:
                if mask[i-1, j] == 1:
                    Jr_face = 0.5 * (Jer[i, j] + Jer[i-1, j])
                    T_up = Te[i, j] if Jr_face > 0.0 else Te[i-1, j]
                    F_r_lh = -alpha * T_up * Jr_face
                else:
                    # Internal Wall -> Zero advection
                    F_r_lh = 0.0
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
                    # Internal Wall -> Zero advection
                    F_z_th = 0.0
            else:
                # Symmetry
                F_z_th = 0.0

            # Bottom face (j-1/2)
            F_z_bh = 0.0
            if j > 0:
                if mask[i, j-1] == 1:
                    Jz_face = 0.5 * (Jez[i, j] + Jez[i, j-1])
                    T_up = Te[i, j] if Jz_face > 0.0 else Te[i, j-1]
                    F_z_bh = -alpha * T_up * Jz_face
                else:
                    # Internal Wall -> Zero advection
                    F_z_bh = 0.0
            else:
                # External Wall -> Zero advection
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