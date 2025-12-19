import numpy as np
from numba import njit

# -------------------------
# Boundary conditions
# -------------------------
@njit(cache=True, fastmath=True)
def apply_no_slip_bc(v, mask):
    """
    Enforce no-slip everywhere required:
      - r=0 (axis) hard Dirichlet
      - outer domain boundaries (r=max, z=min/max)
      - mask==0 (internal no-slip regions)
    """
    Nr, Nz = v.shape

    # r=0 axis is no-slip
    for j in range(Nz):
        v[0, j] = 0.0

    # r = r_max boundary
    for j in range(Nz):
        v[Nr-1, j] = 0.0

    # z boundaries
    for i in range(Nr):
        v[i, 0]    = 0.0
        v[i, Nz-1] = 0.0

    # internal masked cells
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                v[i, j] = 0.0


# -------------------------
# Helpers
# -------------------------
@njit(cache=True)
def _compute_phi_r(phi, dr):
    """
    Radial derivative of phi on a cell-centered grid.
    One-sided at i=0 and i=Nr-1, central elsewhere.
    """
    Nr, Nz = phi.shape
    phi_r = np.empty_like(phi)
    inv2dr = 0.5 / dr
    invdr  = 1.0 / dr

    for i in range(Nr):
        for j in range(Nz):
            if i == 0:
                phi_r[i, j] = (phi[i+1, j] - phi[i, j]) * invdr
            elif i == Nr - 1:
                phi_r[i, j] = (phi[i, j] - phi[i-1, j]) * invdr
            else:
                phi_r[i, j] = (phi[i+1, j] - phi[i-1, j]) * inv2dr
    return phi_r


@njit(cache=True)
def _face_mu(mu_c, mu_nb, nb_mask):
    """Return face-centered viscosity. If neighbor is solid (mask=0), use mu_c."""
    return mu_c if nb_mask == 0 else 0.5 * (mu_c + mu_nb)


# -------------------------
# GS-SOR kernel
# -------------------------
@njit(cache=True)
def _sor_gs_kernel(v, phi_r, Bz, sigma_P, mu, mask, r, dr, dz,
                   omega, max_iters, tol):
    """
    In-place serial Gauss-Seidel SOR for the elliptic problem:
        mu * [ (1/r) d/dr( r dv/dr ) + d2v/dz2 ] - sigma_P * Bz^2 * v = - sigma_P * Bz * dphi/dr

    Strong Dirichlet (no-slip) at:
      - domain boundaries (i==0, i==Nr-1, j==0, j==Nz-1)
      - mask==0 (internal solids)
    """
    Nr, Nz = v.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    for it in range(max_iters):
        max_diff = 0.0

        for i in range(Nr):
            ri = r[i]

            # Dirichlet at domain radial boundaries
            if i == 0 or i == Nr - 1:
                for j in range(Nz):
                    v[i, j] = 0.0
                continue

            # radial face radii
            r_imh = ri - 0.5 * dr
            r_iph = ri + 0.5 * dr
            if r_imh < 0.0:
                r_imh = 0.0  # safety

            for j in range(Nz):
                # Dirichlet at axial boundaries or in solids
                if j == 0 or j == Nz - 1 or mask[i, j] == 0:
                    v[i, j] = 0.0
                    continue

                # Neighbor indices
                im = i - 1
                ip = i + 1
                jm = j - 1
                jp = j + 1

                # Neighbor masks
                mask_imj = mask[im, j]
                mask_ipj = mask[ip, j]
                mask_ijm = mask[i, jm]
                mask_ijp = mask[i, jp]

                # Neighbor values (Dirichlet=0 at boundaries or solids)
                v_imj = 0.0 if mask_imj == 0 else v[im, j]
                v_ipj = 0.0 if mask_ipj == 0 else v[ip, j]
                v_jm  = 0.0 if mask_ijm == 0 else v[i, jm]
                v_jp  = 0.0 if mask_ijp == 0 else v[i, jp]

                # Face-centered viscosities (use mu_c if neighbor is solid)
                mu_c  = mu[i, j]
                mu_rp = _face_mu(mu_c, mu[ip, j], mask_ipj)
                mu_rm = _face_mu(mu_c, mu[im, j], mask_imj)
                mu_zp = _face_mu(mu_c, mu[i, jp], mask_ijp)
                mu_zm = _face_mu(mu_c, mu[i, jm], mask_ijm)

                # Finite-volume coefficients
                # A_rp = mu_rp * (r_{i+1/2}/(r_i * dr^2)), A_rm similar; A_zp = mu_zp/dz^2, A_zm similar
                Arp = mu_rp * (r_iph * inv_dr2) / ri
                Arm = mu_rm * (r_imh * inv_dr2) / ri
                Azp = mu_zp * inv_dz2
                Azm = mu_zm * inv_dz2

                sig = sigma_P[i, j]
                B   = Bz[i, j]
                rhs = sig * B * phi_r[i, j]  # sign consistent with stated PDE

                S = Arp * v_ipj + Arm * v_imj + Azp * v_jp + Azm * v_jm
                C = -(Arp + Arm + Azp + Azm) - sig * B * B

                v_gs  = (S + rhs) / (-C)
                v_old = v[i, j]
                v_new = (1.0 - omega) * v_old + omega * v_gs

                v[i, j] = v_new

                diff = abs(v_new - v_old)
                if diff > max_diff:
                    max_diff = diff

        if max_diff < tol:
            return it + 1, max_diff  # converged

    return max_iters, max_diff  # reached max iterations


# -------------------------
# Residual
# -------------------------
@njit(cache=True)
def _compute_residual_norm(v, phi_r, Bz, sigma_P, mu, mask, r, dr, dz):
    """
    L2 residual norm over interior fluid cells (mask==1, excluding Dirichlet boundaries):
        R = mu*[(1/r)d_r(r d_r v) + d2v/dz2] - sigma_P*Bz^2*v + sigma_P*Bz*phi_r
    """
    Nr, Nz = v.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    res2 = 0.0
    n    = 0

    for i in range(1, Nr - 1):
        if r[i] <= 0.0:
            continue  # safety; with cell-centered grid r[i] > 0
        ri   = r[i]
        r_imh = ri - 0.5 * dr
        r_iph = ri + 0.5 * dr

        for j in range(1, Nz - 1):
            if mask[i, j] == 0:
                continue

            im, ip = i - 1, i + 1
            jm, jp = j - 1, j + 1

            # Neighbor masks
            mask_imj = mask[im, j]
            mask_ipj = mask[ip, j]
            mask_ijm = mask[i, jm]
            mask_ijp = mask[i, jp]

            # Neighbor values (Dirichlet=0 at boundaries or solids)
            v_c   = v[i, j]
            v_imj = 0.0 if mask_imj == 0 else v[im, j]
            v_ipj = 0.0 if mask_ipj == 0 else v[ip, j]
            v_jm  = 0.0 if mask_ijm == 0 else v[i, jm]
            v_jp  = 0.0 if mask_ijp == 0 else v[i, jp]

            # Face-centered viscosities
            mu_c  = mu[i, j]
            mu_rp = _face_mu(mu_c, mu[ip, j], mask_ipj)
            mu_rm = _face_mu(mu_c, mu[im, j], mask_imj)
            mu_zp = _face_mu(mu_c, mu[i, jp], mask_ijp)
            mu_zm = _face_mu(mu_c, mu[i, jm], mask_ijm)

            Arp = mu_rp * (r_iph * inv_dr2) / ri
            Arm = mu_rm * (r_imh * inv_dr2) / ri
            Azp = mu_zp * inv_dz2
            Azm = mu_zm * inv_dz2

            L_v = Arp * (v_ipj - v_c) + Arm * (v_imj - v_c) + Azp * (v_jp - v_c) + Azm * (v_jm - v_c)

            sig = sigma_P[i, j]
            B   = Bz[i, j]
            R   = L_v - sig * B * B * v_c + sig * B * phi_r[i, j]

            res2 += R * R
            n    += 1

    if n == 0:
        return 0.0
    return np.sqrt(res2 / n)


# -------------------------
# Public API
# -------------------------
def solve_vtheta_gs_sor(phi, Bz, sigma_P, mu, dr, dz, mask,
                        omega=1.6, tol=1e-8, max_iters=50_000, v0=None, r=None):
    """
    Serial Gauss-Seidel SOR solver for vtheta with mask-enforced no-slip and
    Dirichlet no-slip on all domain boundaries (including r=0 axis).

    PDE:
        mu * [ (1/r) d/dr( r dv/dr ) + d2v/dz2 ] - sigma_P * Bz^2 * v = - sigma_P * Bz * dphi/dr

    Parameters
    ----------
    phi, Bz, sigma_P, mu : (Nr, Nz) float arrays
        Cell-centered fields.
    dr, dz : float
        Uniform spacings in r and z.
    mask : (Nr, Nz) int array
        1: fluid; 0: solid (no slip).
    omega : float
        SOR relaxation, typically 1.2-1.9. Must be in (0, 2).
    tol : float
        Max-norm update stopping criterion.
    max_iters : int
        Max Gauss-Seidel sweeps.
    v0 : (Nr, Nz) float array or None
        Initial guess. If None, zeros.
    r : (Nr,) float array or None
        Cell-center radii. If None, r[i]=(i+0.5)*dr.

    Returns
    -------
    vtheta : (Nr, Nz) float array
    info   : dict with {'iters','last_update','residual_L2'}
    """
    # Basic checks / shapes
    if not (phi.shape == Bz.shape == sigma_P.shape == mu.shape == mask.shape):
        raise ValueError("All field arrays and mask must have identical (Nr, Nz) shape.")
    Nr, Nz = phi.shape

    if omega <= 0.0 or omega >= 2.0:
        raise ValueError("omega must be in (0, 2) for SOR stability.")

    # Radii
    if r is None:
        r = (np.arange(Nr, dtype=np.float64) + 0.5) * float(dr)
    else:
        r = np.asarray(r, dtype=np.float64)
        if r.shape != (Nr,):
            raise ValueError("r must have shape (Nr,).")
        if r[0] <= 0.0:
            raise ValueError("r[0] must be > 0 (e.g., use cell-centered radii dr/2, 3dr/2, ...).")

    # Working copies (float64)
    v        = np.zeros_like(phi, dtype=np.float64) if v0 is None else np.array(v0, dtype=np.float64, copy=True)
    phi      = np.array(phi, dtype=np.float64, copy=False)
    Bz       = np.array(Bz, dtype=np.float64, copy=False)
    sigma_P  = np.array(sigma_P, dtype=np.float64, copy=False)
    mu       = np.array(mu, dtype=np.float64, copy=False)

    # Precompute dphi/dr
    phi_r = _compute_phi_r(phi, float(dr))

    # Enforce BC on initial guess
    apply_no_slip_bc(v, mask)

    # Iterate
    iters, last_update = _sor_gs_kernel(
        v=v,
        phi_r=phi_r,
        Bz=Bz,
        sigma_P=sigma_P,
        mu=mu,
        mask=mask,
        r=r,
        dr=float(dr),
        dz=float(dz),
        omega=float(omega),
        max_iters=int(max_iters),
        tol=float(tol),
    )

    # One last BC enforcement (safety)
    apply_no_slip_bc(v, mask)

    # Residual
    res = _compute_residual_norm(
        v=v,
        phi_r=phi_r,
        Bz=Bz,
        sigma_P=sigma_P,
        mu=mu,
        mask=mask,
        r=r,
        dr=float(dr),
        dz=float(dz)
    )

    info = {'iters': int(iters), 'last_update': float(last_update), 'residual_L2': float(res)}
    return v, info


# Ions vtheta update kernel using momentum balance given by algebraic solution JxB - Drag = 0
@jit(nopython=True, cache=True)
def update_vtheta_kernel_algebraic(vtheta_out, Jer, Bz, ni, nu_in, un_theta, mask, mi):
    """
    Solves the steady-state algebraic momentum balance for Ion v_theta:
    0 = (J x B) - Drag
    v_theta_i = v_theta_n - (Jr * Bz) / (ni * mi * nu_in)
    
    Parameters:
    -----------
    vtheta_out : 2D array (Nr, Nz) to be updated in-place
    Jer        : 2D array (Nr, Nz), Radial Current Density
    Bz         : 2D array (Nr, Nz), Axial Magnetic Field
    ni         : 2D array (Nr, Nz), Ion Density
    nu_in      : 2D array (Nr, Nz), Ion-Neutral Collision Freq
    un_theta   : 2D array (Nr, Nz), Neutral Gas Velocity
    mask       : 2D array (Nr, Nz), 1=Plasma, 0=Solid
    mi         : float, Ion Mass
    """
    Nr, Nz = Jer.shape
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                # Local scalar values
                n_local = ni[i, j]
                nu_local = nu_in[i, j]
                                    
                # Force calculation
                # Lorentz Force term (assuming J_r x B_z -> -theta direction)
                F_lorentz = -1.0 * Jer[i, j] * Bz[i, j]
                    
                # Drag coefficient = rho_i * nu_in
                drag_coeff = mi * n_local * nu_local
                    
                # Algebraic Solution
                vtheta_out[i, j] = un_theta[i, j] + (F_lorentz / drag_coeff)
                    
            else:
                # Solid boundaries
                vtheta_out[i, j] = 0.0

@jit(nopython=True, cache=True)
def compute_nu_i_kernel(nu_i_out, ni, Ti, nn, Tn, Z, mi, sigma_cx, mask, eps0, q_e, kb):
    """
    Computes total ion collision frequency: nu_i = nu_ii + nu_in
    
    nu_ii (Coulomb): Based on classical Spitzer formula components
    nu_in (Charge Exchange): nn * sigma_cx * v_thermal_rel
    """
    Nr, Nz = nu_i_out.shape
    
    # --- Pre-calculate Physical Constants for nu_ii ---
    # Coefficient for nu_ii = Z^4 * e^4 / (...) * n / T^1.5
    # Standard SI derivation factor
    factor_ii = (Z**4 * q_e**4) / (12.0 * np.pi * np.sqrt(np.pi) * eps0**2 * np.sqrt(mi) * kb**1.5)
    
    # --- Pre-calculate Constants for nu_in ---
    # Relative thermal velocity factor: sqrt( 8*kB / (pi*mi) ) * sqrt(Ti + Tn)
    # Assuming mi approx mn
    factor_in = np.sqrt(8.0 * kb / (np.pi * mi))

    min_T = 1.0 # Avoid division by zero temperature

    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n_local = ni[i, j]
                # Only compute if density is high enough to matter
                if n_local > 1e10:
                    
                    # Local Temps
                    Ti_val = max(Ti[i, j], min_T)
                    Tn_val = max(Tn[i, j], min_T)
                    nn_val = nn[i, j]

                    # 1. Coulomb Collisions (nu_ii)
                    # Calculate Debye Length for Coulomb Logarithm
                    lambda_D = np.sqrt((eps0 * kb * Ti_val) / (n_local * q_e**2))
                    # Classical impact parameter
                    b0 = (Z * q_e**2) / (12.0 * np.pi * eps0 * kb * Ti_val)
                    
                    lnLambda = np.log(lambda_D / b0)
                    if lnLambda < 2.0: lnLambda = 2.0 # Clamp minimum
                    
                    nu_ii = factor_ii * n_local * lnLambda / (Ti_val**1.5)

                    # 2. Charge Exchange (nu_in)
                    v_rel = factor_in * np.sqrt(Ti_val + Tn_val)
                    nu_in = nn_val * sigma_cx * v_rel

                    nu_i_out[i, j] = nu_ii + nu_in
                else:
                    nu_i_out[i, j] = 0.0
            else:
                nu_i_out[i, j] = 0.0

@jit(nopython=True, cache=True)
def compute_beta_i_kernel(beta_i_out, nu_i, Bz, Z, q_e, mi, mask):
    """
    Computes Ion Hall Parameter: beta_i = wci / nu_i
    """
    Nr, Nz = beta_i_out.shape
    gyro_factor = (Z * q_e) / mi
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                nu = nu_i[i, j]
                if nu > 1.0: # Avoid division by tiny numbers
                    wci = gyro_factor * np.abs(Bz[i, j])
                    beta_i_out[i, j] = wci / nu
                else:
                    beta_i_out[i, j] = 0.0
            else:
                beta_i_out[i, j] = 0.0

@jit(nopython=True, cache=True)
def compute_conductivities_kernel(sigma_P, sigma_par, ni, nu_i, beta_i, Z, q_e, mi, mask):
    """
    Computes:
      sigma_parallel = (n * (Ze)^2) / (m * nu)
      sigma_Pedersen = (n * (Ze)^2) / m * (nu / (nu^2 + wci^2))
    """
    Nr, Nz = sigma_P.shape
    prefactor = (Z * q_e)**2 / mi
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n = ni[i, j]
                nu = nu_i[i, j]
                beta = beta_i[i, j]
                
                if nu > 1e-5:
                    # Parallel Conductivity
                    s_par = (prefactor * n) / nu
                    sigma_par[i, j] = s_par
                    
                    # Pedersen Conductivity
                    # Relation: sigma_P = sigma_par / (1 + beta^2)
                    sigma_P[i, j] = s_par / (1.0 + beta**2)
                else:
                    sigma_par[i, j] = 0.0
                    sigma_P[i, j] = 0.0
            else:
                sigma_par[i, j] = 0.0
                sigma_P[i, j] = 0.0

@jit(nopython=True, cache=True)
def update_Ti_joule_heating_kernel(Ti_out, Tn, Te, 
                                   Jer, Jez,            # Currents
                                   sigma_P, sigma_par,  # Conductivities
                                   ni, nu_in, mi, mn, mask, kb, eps0, q_e, Z):
    """
    Updates Ion Temperature (Ti) using explicit Joule Heating (J*E) as the source.
    
    Balance:
    (J_perp^2 / sigma_P) + (J_par^2 / sigma_par) + Q_ie = Q_in_thermal
    
    where Q_in_thermal = 3 * (mi/mn) * ni * nu_in * kb * (Ti - Tn)
    """
    Nr, Nz = Ti_out.shape
    
    me = constants.m_e
    # Constant block for nu_ei (Electron-Ion)
    factor_ei = (Z**2 * q_e**4) / (12.0 * np.pi * np.sqrt(np.pi) * eps0**2 * np.sqrt(me) * kb**1.5)
    
    ratio_me_mi = me / mi
    ratio_mi_mn = mi / mn

    min_sigma = 1e-6 # Avoid division by zero

    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n_local = ni[i, j]
                
                if n_local > 1e10:
                    # --- 1. Calculate Joule Heating (Source) ---
                    # Assuming Jer is perpendicular (radial) and Jez is parallel (axial)
                    # NOTE: If you have Br, strictly speaking Jez has a perp component,
                    # but for typical Bz >> Br, this separation holds.
                    
                    s_p = max(sigma_P[i, j], min_sigma)
                    s_par = max(sigma_par[i, j], min_sigma)
                    
                    # Q_joule = J_perp^2 / sigma_P + J_par^2 / sigma_par
                    # Result is in Watts/m^3
                    Q_joule = (Jer[i, j]**2 / s_p) + (Jez[i, j]**2 / s_par)

                    # --- 2. Electron-Ion Heat Transfer (Source/Sink) ---
                    Te_local = max(Te[i, j], 0.1)
                    Tn_local = Tn[i, j]
                    
                    # Calculate nu_ei
                    lambda_D = np.sqrt((eps0 * kb * Te_local) / (n_local * q_e**2))
                    b0 = (Z * q_e**2) / (12.0 * np.pi * eps0 * kb * Te_local)
                    lnLambda = max(2.0, np.log(lambda_D / b0))
                    nu_ei = factor_ei * n_local * lnLambda / (Te_local**1.5)

                    # Q_ie coeff: A * (Te - Ti)
                    # A = 3 * (me/mi) * n * nu_ei * kb
                    A_coeff = 3.0 * ratio_me_mi * n_local * nu_ei * kb
                    
                    # --- 3. Neutral Cooling (Sink) ---
                    # Q_in coeff: B * (Ti - Tn)
                    # B = 3 * (mi/mn) * n * nu_in * kb
                    # Note: We use only the thermal relaxation part here
                    nu_in_local = nu_in[i, j]
                    B_coeff = 3.0 * ratio_mi_mn * n_local * nu_in_local * kb
                    
                    # --- 4. Solve Balance ---
                    # Q_joule + A(Te - Ti) = B(Ti - Tn)
                    # Q_joule + A*Te + B*Tn = (A + B) * Ti
                    
                    denom = A_coeff + B_coeff
                    if denom > 1e-12:
                        Ti_new = (Q_joule + A_coeff * Te_local + B_coeff * Tn_local) / denom
                        Ti_out[i, j] = Ti_new
                    else:
                        Ti_out[i, j] = Tn_local
                else:
                    Ti_out[i, j] = Tn[i, j]
            else:
                Ti_out[i, j] = 300.0