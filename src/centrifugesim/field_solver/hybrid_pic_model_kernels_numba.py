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
    sigma_en_m2=5e-20,     # momentum-transfer X-section (tune for your gas)
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
    Te, ne, nn, Br, Bz,
    lnLambda=10.0,
    sigma_en_m2=5e-20,
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
    Bmag = np.sqrt(Br*Br + Bz*Bz) + B_FLOOR

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


@numba.njit(inline='always')
def _face_avg_center(val_c, val_nb, nb_is_active):
    """Average to neighbor only if that neighbor is active (mask==1)."""
    return 0.5 * (val_c + val_nb) if nb_is_active else val_c


@numba.njit(inline='always')
def _face_sigma_center(sig_c, sig_nb, nb_is_active):
    """Face conductivity: average only across active neighbor; otherwise one-sided."""
    return 0.5 * (sig_c + sig_nb) if nb_is_active else sig_c


@numba.njit(cache=True)
def _grad_rz_center(field, dr, dz, r, mask):
    """Masked, axis-safe centered gradients at cell centers."""
    Nr, Nz = field.shape
    gr = np.zeros_like(field)
    gz = np.zeros_like(field)

    for i in numba.prange(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                gr[i, j] = 0.0
                gz[i, j] = 0.0
                continue

            # d()/dr
            if i == 0:  # axis
                if mask[i+1, j] == 1:
                    gr[i, j] = (field[i+1, j] - field[i, j]) / dr
                else:
                    gr[i, j] = 0.0
            elif i == Nr - 1:  # outer wall
                if mask[i-1, j] == 1:
                    gr[i, j] = (field[i, j] - field[i-1, j]) / dr
                else:
                    gr[i, j] = 0.0
            else:
                left = (mask[i-1, j] == 1)
                right = (mask[i+1, j] == 1)
                if left and right:
                    gr[i, j] = (field[i+1, j] - field[i-1, j]) / (2.0*dr)
                elif right:
                    gr[i, j] = (field[i+1, j] - field[i, j]) / dr
                elif left:
                    gr[i, j] = (field[i, j] - field[i-1, j]) / dr
                else:
                    gr[i, j] = 0.0

            # d()/dz
            if j == 0:
                if mask[i, j+1] == 1:
                    gz[i, j] = (field[i, j+1] - field[i, j]) / dz
                else:
                    gz[i, j] = 0.0
            elif j == Nz - 1:
                if mask[i, j-1] == 1:
                    gz[i, j] = (field[i, j] - field[i, j-1]) / dz
                else:
                    gz[i, j] = 0.0
            else:
                down = (mask[i, j-1] == 1)
                up   = (mask[i, j+1] == 1)
                if down and up:
                    gz[i, j] = (field[i, j+1] - field[i, j-1]) / (2.0*dz)
                elif up:
                    gz[i, j] = (field[i, j+1] - field[i, j]) / dz
                elif down:
                    gz[i, j] = (field[i, j] - field[i, j-1]) / dz
                else:
                    gz[i, j] = 0.0

    return gr, gz


@numba.njit(cache=True)
def compute_electric_field(phi, dr, dz, r, mask):
    """Returns Er, Ez at centers, with E = -∇phi (mask-aware, axis safe)."""
    dphi_dr, dphi_dz = _grad_rz_center(phi, dr, dz, r, mask)
    Er = -dphi_dr
    Ez = -dphi_dz
    return Er, Ez


@numba.njit(cache=True)
def compute_currents(phi, sigma_P, sigma_H, sigma_parallel,
                     pe, Bz, un_theta, un_r, ne,
                     dr, dz, r, e_charge, mask):
    """
    Current density components Jr, Jz at centers using the model.
    Also returns the source terms Sr, Sz for diagnostics.
    """
    # Gradients
    dphi_dr, dphi_dz = _grad_rz_center(phi, dr, dz, r, mask)
    dpe_dr,  dpe_dz  = _grad_rz_center(pe,  dr, dz, r, mask)

    tiny = 1e-300
    inv_en = 1.0 / (e_charge * (ne + tiny))

    # Sources
    Sr = -inv_en * dpe_dr + Bz * un_theta
    Sz = -inv_en * dpe_dz

    # Currents (E = -∇phi)
    Jr = -sigma_P * dphi_dr + sigma_P * Sr + sigma_H * Bz * un_r
    Jz = -sigma_parallel * dphi_dz + sigma_parallel * Sz

    # Clean masked cells
    Nr, Nz = phi.shape
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                Jr[i, j] = 0.0
                Jz[i, j] = 0.0
                Sr[i, j] = 0.0
                Sz[i, j] = 0.0

    return Jr, Jz, Sr, Sz


@numba.njit(cache=True)
def joule_heating(
    phi, sigma_P, sigma_H, sigma_parallel,
    pe, Bz, un_theta, un_r, ne,
    dr, dz, r, e_charge, mask
):
    """
    Volumetric power deposition by Joule heating.

    Returns
    -------
    q_ohm : np.ndarray
        Non-negative irreversible heating:
        q_ohm = sigma_P*(E_r + S_r)^2 + sigma_parallel*(E_z + S_z)^2.
    q_raw : np.ndarray
        Diagnostic raw power q_raw = J · E = J_r*E_r + J_z*E_z (may be negative locally).
    """
    Er, Ez = compute_electric_field(phi, dr, dz, r, mask)
    Jr, Jz, Sr, Sz = compute_currents(
        phi, sigma_P, sigma_H, sigma_parallel,
        pe, Bz, un_theta, un_r, ne,
        dr, dz, r, e_charge, mask
    )

    q_raw = Jr * Er + Jz * Ez
    q_ohm = sigma_P * (Er + Sr) * (Er + Sr) + sigma_parallel * (Ez + Sz) * (Ez + Sz)

    Nr, Nz = phi.shape
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                q_ohm[i, j] = 0.0
                q_raw[i, j] = 0.0

    return q_ohm, q_raw


@numba.njit(cache=True)
def divergence_cyl(Jr, Jz, r, dr, dz, mask):
    """
    ∇·J in axisymmetric (r–z): (1/r) ∂_r(r Jr) + ∂_z(Jz) using FV faces.
    """
    Nr, Nz = Jr.shape
    div = np.zeros_like(Jr)

    for i in range(Nr):
        for j in range(1, Nz-1):
            if mask[i, j] == 0:
                continue

            # Radial part
            if i == 0:
                J_flux_r_p = 0.5 * (Jr[i+1, j] + Jr[i, j])
                div_r = 2.0 * J_flux_r_p / dr
            elif i < Nr - 1:
                r_p = r[i] + 0.5 * dr
                r_m = r[i] - 0.5 * dr
                J_flux_r_p = 0.5 * (Jr[i+1, j] + Jr[i, j])
                J_flux_r_m = 0.5 * (Jr[i, j] + Jr[i-1, j])
                div_r = (r_p * J_flux_r_p - r_m * J_flux_r_m) / (r[i] * dr)
            else:
                continue

            # Axial part
            J_flux_z_p = 0.5 * (Jz[i, j+1] + Jz[i, j])
            J_flux_z_m = 0.5 * (Jz[i, j] + Jz[i, j-1])
            div_z = (J_flux_z_p - J_flux_z_m) / dz

            div[i, j] = div_r + div_z

    return div


#################################################################################
############################## Elliptic solver core #############################
#################################################################################
@numba.njit(cache=True, parallel=True)
def _solve_phi_core(
    phi, sigma_P, sigma_H, sigma_parallel, pe, Bz, un_theta, un_r, ne,
    r, dr, dz, e, max_iter, tol, omega, mask
):
    """
    Core JIT-compiled Red-Black SOR solver for phi (mask-aware, Neumann walls).
    """
    Nr, Nz = phi.shape

    # --- 1) Build sources and J_S = sigma * S ---
    grad_pe_r = np.zeros_like(phi)
    grad_pe_z = np.zeros_like(phi)

    for i in numba.prange(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                grad_pe_r[i, j] = 0.0
                grad_pe_z[i, j] = 0.0
                continue

            # d(pe)/dr
            if i == 0:
                grad_pe_r[i, j] = (pe[i+1, j] - pe[i, j]) / dr if mask[i+1, j] == 1 else 0.0
            elif i == Nr - 1:
                grad_pe_r[i, j] = (pe[i, j] - pe[i-1, j]) / dr if mask[i-1, j] == 1 else 0.0
            else:
                L = (mask[i-1, j] == 1); R = (mask[i+1, j] == 1)
                if L and R:
                    grad_pe_r[i, j] = (pe[i+1, j] - pe[i-1, j]) / (2.0*dr)
                elif R:
                    grad_pe_r[i, j] = (pe[i+1, j] - pe[i, j]) / dr
                elif L:
                    grad_pe_r[i, j] = (pe[i, j] - pe[i-1, j]) / dr
                else:
                    grad_pe_r[i, j] = 0.0

            # d(pe)/dz
            if j == 0:
                grad_pe_z[i, j] = (pe[i, j+1] - pe[i, j]) / dz if mask[i, j+1] == 1 else 0.0
            elif j == Nz - 1:
                grad_pe_z[i, j] = (pe[i, j] - pe[i, j-1]) / dz if mask[i, j-1] == 1 else 0.0
            else:
                D = (mask[i, j-1] == 1); U = (mask[i, j+1] == 1)
                if D and U:
                    grad_pe_z[i, j] = (pe[i, j+1] - pe[i, j-1]) / (2.0*dz)
                elif U:
                    grad_pe_z[i, j] = (pe[i, j+1] - pe[i, j]) / dz
                elif D:
                    grad_pe_z[i, j] = (pe[i, j] - pe[i, j-1]) / dz
                else:
                    grad_pe_z[i, j] = 0.0

    tiny = 1e-300
    one_over_ene = 1.0 / (e * (ne + tiny))
    Sr = -one_over_ene * grad_pe_r + Bz * un_theta
    Sz = -one_over_ene * grad_pe_z

    J_S_r = sigma_P * Sr + sigma_H * Bz * un_r
    J_S_z = sigma_parallel * Sz

    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                Sr[i, j] = 0.0; Sz[i, j] = 0.0
                J_S_r[i, j] = 0.0; J_S_z[i, j] = 0.0

    # --- 2) RHS = div(J_S) ---
    RHS = np.zeros_like(phi)
    for i in range(Nr):
        for j in range(1, Nz - 1):
            if mask[i, j] == 0:
                continue

            # radial divergence
            if i == 0:
                nb_p = (mask[i+1, j] == 1)
                J_flux_r_p = _face_avg_center(J_S_r[i, j], J_S_r[i+1, j], nb_p)
                div_r = 2.0 * J_flux_r_p / dr
            elif i < Nr - 1:
                r_p = r[i] + 0.5 * dr
                r_m = r[i] - 0.5 * dr
                nb_p = (mask[i+1, j] == 1)
                nb_m = (mask[i-1, j] == 1)
                J_flux_r_p = _face_avg_center(J_S_r[i, j], J_S_r[i+1, j], nb_p)
                J_flux_r_m = _face_avg_center(J_S_r[i, j], J_S_r[i-1, j], nb_m)
                div_r = (r_p * J_flux_r_p - r_m * J_flux_r_m) / (r[i] * dr)
            else:
                continue

            # axial divergence
            nb_up = (mask[i, j+1] == 1)
            nb_dn = (mask[i, j-1] == 1)
            J_flux_z_p = _face_avg_center(J_S_z[i, j], J_S_z[i, j+1], nb_up)
            J_flux_z_m = _face_avg_center(J_S_z[i, j], J_S_z[i, j-1], nb_dn)
            div_z = (J_flux_z_p - J_flux_z_m) / dz

            RHS[i, j] = div_r + div_z

    # --- 3) Red-Black SOR ---
    phi_old = np.copy(phi)

    for k in range(max_iter):
        phi_old[:] = phi[:]

        # RED
        for i in numba.prange(1, Nr - 1):
            for j in range(1, Nz - 1):
                if ((i + j) % 2 == 0) and (mask[i, j] == 1):
                    r_p = r[i] + 0.5 * dr; r_m = r[i] - 0.5 * dr

                    nb_rp = (mask[i+1, j] == 1)
                    nb_rm = (mask[i-1, j] == 1)
                    sig_rp = _face_sigma_center(sigma_P[i, j], sigma_P[i+1, j], nb_rp)
                    sig_rm = _face_sigma_center(sigma_P[i, j], sigma_P[i-1, j], nb_rm)
                    A = (r_p * sig_rp) / (r[i] * dr**2)
                    B = (r_m * sig_rm) / (r[i] * dr**2)

                    nb_zp = (mask[i, j+1] == 1)
                    nb_zm = (mask[i, j-1] == 1)
                    sig_zp = _face_sigma_center(sigma_parallel[i, j], sigma_parallel[i, j+1], nb_zp)
                    sig_zm = _face_sigma_center(sigma_parallel[i, j], sigma_parallel[i, j-1], nb_zm)
                    C = sig_zp / dz**2
                    D = sig_zm / dz**2

                    diag = A + B + C + D
                    if diag == 0.0:
                        continue
                    num = (A * phi_old[i+1, j] + B * phi_old[i-1, j] +
                           C * phi_old[i, j+1] + D * phi_old[i, j-1] - RHS[i, j])
                    phi_gs = num / diag
                    phi[i, j] = phi_old[i, j] + omega * (phi_gs - phi_old[i, j])

        # BLACK
        for i in numba.prange(1, Nr - 1):
            for j in range(1, Nz - 1):
                if ((i + j) % 2 != 0) and (mask[i, j] == 1):
                    r_p = r[i] + 0.5 * dr; r_m = r[i] - 0.5 * dr

                    nb_rp = (mask[i+1, j] == 1)
                    nb_rm = (mask[i-1, j] == 1)
                    sig_rp = _face_sigma_center(sigma_P[i, j], sigma_P[i+1, j], nb_rp)
                    sig_rm = _face_sigma_center(sigma_P[i, j], sigma_P[i-1, j], nb_rm)
                    A = (r_p * sig_rp) / (r[i] * dr**2)
                    B = (r_m * sig_rm) / (r[i] * dr**2)

                    nb_zp = (mask[i, j+1] == 1)
                    nb_zm = (mask[i, j-1] == 1)
                    sig_zp = _face_sigma_center(sigma_parallel[i, j], sigma_parallel[i, j+1], nb_zp)
                    sig_zm = _face_sigma_center(sigma_parallel[i, j], sigma_parallel[i, j-1], nb_zm)
                    C = sig_zp / dz**2
                    D = sig_zm / dz**2

                    diag = A + B + C + D
                    if diag == 0.0:
                        continue
                    num = (A * phi[i+1, j] + B * phi[i-1, j] +
                           C * phi[i, j+1] + D * phi[i, j-1] - RHS[i, j])
                    phi_gs = num / diag
                    phi[i, j] = phi_old[i, j] + omega * (phi_gs - phi_old[i, j])

        # Axis r=0 (even/odd j)
        for j in numba.prange(1, Nz-1):
            if (j % 2 == 0) and (mask[0, j] == 1):
                A0 = 4.0 * sigma_P[0, j] / dr**2
                nb_zp = (mask[0, j+1] == 1); nb_zm = (mask[0, j-1] == 1)
                sig_zp = _face_sigma_center(sigma_parallel[0, j], sigma_parallel[0, j+1], nb_zp)
                sig_zm = _face_sigma_center(sigma_parallel[0, j], sigma_parallel[0, j-1], nb_zm)
                C0 = sig_zp / dz**2; D0 = sig_zm / dz**2
                diag0 = A0 + C0 + D0
                if diag0 != 0.0:
                    num0 = (A0 * phi_old[1, j] + C0 * phi_old[0, j+1] + D0 * phi_old[0, j-1] - RHS[0, j])
                    phi0 = num0 / diag0
                    phi[0, j] = phi_old[0, j] + omega * (phi0 - phi_old[0, j])

        for j in numba.prange(1, Nz-1):
            if (j % 2 != 0) and (mask[0, j] == 1):
                A0 = 4.0 * sigma_P[0, j] / dr**2
                nb_zp = (mask[0, j+1] == 1); nb_zm = (mask[0, j-1] == 1)
                sig_zp = _face_sigma_center(sigma_parallel[0, j], sigma_parallel[0, j+1], nb_zp)
                sig_zm = _face_sigma_center(sigma_parallel[0, j], sigma_parallel[0, j-1], nb_zm)
                C0 = sig_zp / dz**2; D0 = sig_zm / dz**2
                diag0 = A0 + C0 + D0
                if diag0 != 0.0:
                    num0 = (A0 * phi[1, j] + C0 * phi[0, j+1] + D0 * phi[0, j-1] - RHS[0, j])
                    phi0 = num0 / diag0
                    phi[0, j] = phi_old[0, j] + omega * (phi0 - phi_old[0, j])

        # Neumann walls each sweep
        # (A) z = L (top): Jz=0 -> dphi/dz = Sz
        for i in numba.prange(Nr):
            if mask[i, Nz-1] == 1:
                phi[i, Nz-1] = phi[i, Nz-2] + dz * Sz[i, Nz-1]

        # (B) r = R (outer, non-anode): Jr=0 -> dphi/dr = Sr + (sigma_H/sigma_P) Bz un_r
        tiny_sig = 1e-30
        for j in numba.prange(Nz):
            if mask[Nr-1, j] == 1:
                sigP = sigma_P[Nr-1, j]
                hall_ratio = sigma_H[Nr-1, j] / (sigP + tiny_sig)
                dphi_dr = Sr[Nr-1, j] + hall_ratio * Bz[Nr-1, j] * un_r[Nr-1, j]
                phi[Nr-1, j] = phi[Nr-2, j] + dr * dphi_dr

        # Convergence check
        if (k % 5) == 0:
            norm_phi = 0.0; err = 0.0
            for i in range(Nr):
                for j in range(Nz):
                    v = phi_old[i, j]
                    dv = phi[i, j] - v
                    norm_phi += v*v
                    err += dv*dv
            if norm_phi == 0.0:
                norm_phi = 1.0 if err > 0.0 else 0.0
            rel = (err / norm_phi)**0.5
            if rel < tol:
                return phi, k + 1

    return phi, max_iter


__all__ = [
    "_kBT",
    "electron_collision_frequencies",
    "electron_conductivities",
    "_face_avg_center",
    "_face_sigma_center",
    "_grad_rz_center",
    "compute_electric_field",
    "compute_currents",
    "joule_heating",
    "divergence_cyl",
    "_solve_phi_core",
]