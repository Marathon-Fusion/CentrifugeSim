import numpy as np
import numba

from centrifugesim import constants

@numba.jit(nopython=True, parallel=True, nogil=True)
def solve_step(Te, Te_new, dr, dz, r_vec, n_e, Q_Joule,
               br, bz, kappa_parallel, kappa_perp,
               mask, dt):
    """
    Advances the electron temperature Te by one time step DT.
    Uses an explicit, cell-centered finite difference scheme.
    The energy equation is only solved where mask == 1.
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