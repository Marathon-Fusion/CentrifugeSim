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


@njit(cache=True)
def electron_collision_frequencies(
    Te, ne, nn,
    lnLambda=10.0,
    sigma_en_m2=3e-19, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
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
    Te, ne, Bmag, nu_e,
    lnLambda=10.0,
    sigma_en_m2=3e-19, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
    Te_is_eV=False
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
    Bmag += B_FLOOR

    # Electron gyrofrequency magnitude
    Omega_e = constants.q_e * Bmag / constants.m_e  # rad/s

    # Prefactor ne e^2 / m_e
    pref = ne * (constants.q_e * constants.q_e) / constants.m_e  # S·s/m

    # Components (electrons only)
    sigma_par_e = pref / nu_e
    denom = (nu_e*nu_e + Omega_e*Omega_e)
    sigma_P_e = pref * (nu_e / denom)
    sigma_H_e = - pref * (Omega_e / denom)

    return sigma_par_e, sigma_P_e, sigma_H_e, Omega_e/nu_e  # β_e

# Update solver !
@numba.jit(nopython=True, parallel=True, nogil=True)
def solve_step(Te, Te_new, dr, dz, r_vec, n_e, Q_Joule,
               br, bz, kappa_parallel, kappa_perp,
               Jer, Jez,
               mask, dt):
    """
    Advances electron temperature one explicit step with anisotropic conduction,
    but with *no* heat flux across faces that touch masked cells (mask==0)
    or the outer domain boundary. This prevents sinks at electrodes/boundaries.

    All arrays are shape (NR, NZ). mask is int8 with 1=solve region, 0=masked.
    Explicit Euler for now just for testing, do not use for production runs.
    TO DO:
        USE ne_floor instead of "tiny"
        Change to Douglas-ADI to get unconditionally stable behavior and larger timestep size.
    """
    NR, NZ = Te.shape
    kb = constants.kb
    qe = constants.q_e
    tiny = 1e9  # avoid division by zero

    for i in numba.prange(1, NR - 1):
        for j in range(1, NZ - 1):

            if not (mask[i, j] == 1 and n_e[i, j] > 0.0):
                # Leave Te_new untouched for masked/vacuum cells
                continue

            # --- Right face (i+1/2, j) ---
            qr_rh = 0.0
            if i < NR - 1 and mask[i+1, j] == 1:
                # Ensure the cross-term stencil does not sample masked cells
                if (mask[i, j+1] == 1 and mask[i, j-1] == 1 and
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
                # else: face touches masked cells via cross-stencil → enforce qr_rh = 0

            # --- Left face (i-1/2, j) ---
            qr_lh = 0.0
            # Symmetry at axis: enforce zero-flux at the inner boundary
            if i > 1 and mask[i-1, j] == 1:
                if (mask[i, j+1] == 1 and mask[i, j-1] == 1 and
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
                # else: face touches masked cells via cross-stencil → enforce qr_lh = 0
            # If i == 1, we are at the first active ring next to the axis: keep qr_lh = 0

            # --- Top face (i, j+1/2) ---
            qz_th = 0.0
            if j < NZ - 1 and mask[i, j+1] == 1:
                if (mask[i+1, j] == 1 and mask[i-1, j] == 1 and
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
                # else: face touches masked cells via cross-stencil → enforce qz_th = 0

            # --- Bottom face (i, j-1/2) ---
            qz_bh = 0.0
            if j > 1 and mask[i, j-1] == 1:
                if (mask[i+1, j] == 1 and mask[i-1, j] == 1 and
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
                # else: face touches masked cells via cross-stencil → enforce qz_bh = 0
            # If j == 1, we’re at the first active row next to z-min: keep qz_bh = 0

            # --- Divergence (finite-volume consistent in RZ) ---
            r_center = r_vec[i] + 1e-12
            r_rh_face = r_vec[i] + 0.5 * dr
            r_lh_face = r_vec[i] - 0.5 * dr

            div_qr_term = (r_rh_face * qr_rh - r_lh_face * qr_lh) / (r_center * dr)
            div_qz_term = (qz_th - qz_bh) / dz
            div_q = div_qr_term + div_qz_term

            # =========================
            # Enthalpy advection (conservative (J,T) form, upwind T only)
            # =========================
            # Right face i+1/2 (no flux if neighbor masked)
            F_r_rh = 0.0
            if i < NR - 1 and mask[i+1, j] == 1:
                Jr_face = 0.5 * (Jer[i, j] + Jer[i+1, j])
                if Jr_face > 0.0:
                    T_up = Te[i, j]
                else:
                    T_up = Te[i+1, j]
                F_r_rh = -(2.5 * kb / qe) * T_up * Jr_face

            # Left face i-1/2
            F_r_lh = 0.0
            if i > 1 and mask[i-1, j] == 1:
                Jr_face = 0.5 * (Jer[i, j] + Jer[i-1, j])
                if Jr_face > 0.0:
                    T_up = Te[i-1, j]
                else:
                    T_up = Te[i, j]
                F_r_lh = -(2.5 * kb / qe) * T_up * Jr_face

            # Top face j+1/2
            F_z_th = 0.0
            if j < NZ - 1 and mask[i, j+1] == 1:
                Jz_face = 0.5 * (Jez[i, j] + Jez[i, j+1])
                if Jz_face > 0.0:
                    T_up = Te[i, j]
                else:
                    T_up = Te[i, j+1]
                F_z_th = -(2.5 * kb / qe) * T_up * Jz_face

            # Bottom face j-1/2
            F_z_bh = 0.0
            if j > 1 and mask[i, j-1] == 1:
                Jz_face = 0.5 * (Jez[i, j] + Jez[i, j-1])
                if Jz_face > 0.0:
                    T_up = Te[i, j-1]
                else:
                    T_up = Te[i, j]
                F_z_bh = -(2.5 * kb / qe) * T_up * Jz_face

            # Divergence of advective enthalpy flux in RZ
            div_Fadv = (r_rh_face * F_r_rh - r_lh_face * F_r_lh) / (r_center * dr) + (F_z_th - F_z_bh) / dz

            # -----------------------------------
            # TO DO:
            #      Change this to BC that accounts for Jez and kappa_parallel at cathode exit plane
            # Corrections for masked faces (no-flux enforced, so add back missing fluxes)
            
            S_mask = 0.0
            # radial corrections (RZ weighting)
            if mask[i+1, j] == 0:  # right face missing
                S_mask += (2.5 * kb / qe) * Te[i, j] * ( (r_rh_face / (r_center * dr)) * Jer[i, j] )
            if mask[i-1, j] == 0 and i > 1:  # left face missing
                S_mask -= (2.5 * kb / qe) * Te[i, j] * ( (r_lh_face / (r_center * dr)) * Jer[i, j] )

            # axial corrections
            if mask[i, j+1] == 0:  # top face missing
                S_mask += (2.5 * kb / qe) * Te[i, j] * ( Jez[i, j] / dz )
            if mask[i, j-1] == 0 and j > 1:  # bottom face missing
                S_mask -= (2.5 * kb / qe) * Te[i, j] * ( Jez[i, j] / dz )

            # =========================
            # Update
            # =========================
            rhs = -div_q - div_Fadv + Q_Joule[i, j] + S_mask 
            dTe_dt = (2.0 / (3.0 * n_e[i, j] * kb)) * rhs
            Te_new[i, j] = Te[i, j] + dt * dTe_dt