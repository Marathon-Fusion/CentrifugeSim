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
    sigma_en_m2=5e-20, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
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
    nu_ei = c_num * ne / (c_den * (kBT**1.5 + 0.0))            # 1/s

    # Total + floors
    nu_e  = nu_en + nu_ei
    nu_en = np.maximum(nu_en, NU_FLOOR)
    nu_ei = np.maximum(nu_ei, NU_FLOOR)
    nu_e  = np.maximum(nu_e,  NU_FLOOR)
    return nu_en, nu_ei, nu_e


@njit(cache=True)
def electron_conductivities(
    Te, ne, nn, Bmag,
    lnLambda=10.0,
    sigma_en_m2=5e-20, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
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
    # Collisions
    _, _, nu_e = electron_collision_frequencies(Te, ne, nn, lnLambda, sigma_en_m2, Te_is_eV)

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

# Move this solve to 
@numba.jit(nopython=True, parallel=True, nogil=True)
def solve_step(Te, Te_new, dr, dz, r_vec, n_e, Q_Joule,
               br, bz, kappa_parallel, kappa_perp,
               mask, dt):
    """
    Advances the electron temperature Te by one time step DT.
    Uses an explicit, cell-centered finite difference scheme.
    The energy equation is only solved where mask == 1.
    Should improve this to reduce number of steps needed, explicit Euler for now
    just for testing, do not use for production runs.
    """
    NR, NZ = Te.shape

    # Loop over the interior of the grid (parallelized by Numba)
    for i in numba.prange(1, NR - 1):
        for j in numba.prange(1, NZ - 1):
            
            # Only solve the equation inside the plasma region (mask == 1)
            if(mask[i, j] == 1 and n_e[i, j]>0):
                # --- STABLE DIVERGENCE CALCULATION (CELL-CENTERED FLUX) ---
                # This correctly calculates flux from plasma cells into the fixed-T cathode cells
                # 1. Calculate flux q_r at the right-hand face (i + 1/2, j)
                br_rh = (br[i, j] + br[i+1, j]) / 2.0
                bz_rh = (bz[i, j] + bz[i+1, j]) / 2.0
                k_par_rh = (kappa_parallel[i, j] + kappa_parallel[i+1, j]) / 2.0
                k_perp_rh = (kappa_perp[i, j] + kappa_perp[i+1, j]) / 2.0
                k_a_rh = k_par_rh - k_perp_rh
                k_rr_rh = k_perp_rh + k_a_rh * br_rh**2
                k_rz_rh = k_a_rh * br_rh * bz_rh
                dT_dr_rh = (Te[i+1, j] - Te[i, j]) / dr
                dT_dz_rh = (Te[i, j+1] - Te[i, j-1] + Te[i+1, j+1] - Te[i+1, j-1]) / (4 * dz)
                qr_rh = -(k_rr_rh * dT_dr_rh + k_rz_rh * dT_dz_rh)

                # 2. Calculate flux q_r at the left-hand face (i - 1/2, j)
                br_lh = (br[i, j] + br[i-1, j]) / 2.0
                bz_lh = (bz[i, j] + bz[i-1, j]) / 2.0
                k_par_lh = (kappa_parallel[i, j] + kappa_parallel[i-1, j]) / 2.0
                k_perp_lh = (kappa_perp[i, j] + kappa_perp[i-1, j]) / 2.0
                k_a_lh = k_par_lh - k_perp_lh
                k_rr_lh = k_perp_lh + k_a_lh * br_lh**2
                k_rz_lh = k_a_lh * br_lh * bz_lh
                dT_dr_lh = (Te[i, j] - Te[i-1, j]) / dr
                dT_dz_lh = (Te[i, j+1] - Te[i, j-1] + Te[i-1, j+1] - Te[i-1, j-1]) / (4 * dz)
                qr_lh = -(k_rr_lh * dT_dr_lh + k_rz_lh * dT_dz_lh)

                # 3. Calculate flux q_z at the top face (i, j + 1/2)
                br_th = (br[i, j] + br[i, j+1]) / 2.0
                bz_th = (bz[i, j] + bz[i, j+1]) / 2.0
                k_par_th = (kappa_parallel[i, j] + kappa_parallel[i, j+1]) / 2.0
                k_perp_th = (kappa_perp[i, j] + kappa_perp[i+1, j]) / 2.0
                k_a_th = k_par_th - k_perp_th
                k_zz_th = k_perp_th + k_a_th * bz_th**2
                k_rz_th = k_a_th * br_th * bz_th
                dT_dz_th = (Te[i, j+1] - Te[i, j]) / dz
                dT_dr_th = (Te[i+1, j] - Te[i-1, j] + Te[i+1, j+1] - Te[i-1, j+1]) / (4 * dr)
                qz_th = -(k_rz_th * dT_dr_th + k_zz_th * dT_dz_th)

                # 4. Calculate flux q_z at the bottom face (i, j - 1/2)
                br_bh = (br[i, j] + br[i, j-1]) / 2.0
                bz_bh = (bz[i, j] + bz[i, j-1]) / 2.0
                k_par_bh = (kappa_parallel[i, j] + kappa_parallel[i, j-1]) / 2.0
                k_perp_bh = (kappa_perp[i, j] + kappa_perp[i, j-1]) / 2.0
                k_a_bh = k_par_bh - k_perp_bh
                k_zz_bh = k_perp_bh + k_a_bh * bz_bh**2
                k_rz_bh = k_a_bh * br_bh * bz_bh
                dT_dz_bh = (Te[i, j] - Te[i, j-1]) / dz
                dT_dr_bh = (Te[i+1, j] - Te[i-1, j] + Te[i+1, j-1] - Te[i-1, j-1]) / (4 * dr)
                qz_bh = -(k_rz_bh * dT_dr_bh + k_zz_bh * dT_dz_bh)

                # 5. Calculate divergence using the face fluxes
                r_rh_face = r_vec[i] + dr / 2.0
                r_lh_face = r_vec[i] - dr / 2.0
                div_qr_term = (r_rh_face * qr_rh - r_lh_face * qr_lh) / ((r_vec[i] + 1e-12) * dr)
                div_qz_term = (qz_th - qz_bh) / dz
                div_q = div_qr_term + div_qz_term
                
                # --- Time Update (Forward Euler) ---
                rhs = -div_q + Q_Joule
                dTe_dt = (2.0 / (3.0 * n_e[i, j] * constants.kb)) * rhs
                
                Te_new[i, j] = Te[i, j] + dt * dTe_dt

    return Te_new