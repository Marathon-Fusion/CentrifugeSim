import numpy as np
from numba import njit

from centrifugesim import constants

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
@njit(cache=True)
def update_vtheta_kernel_algebraic(vtheta_out, Jr, Bz, ni, nu_in, un_theta, mask, mi):
    """
    Solves the steady-state algebraic momentum balance for Ion v_theta:
    0 = (J x B) - Drag
    v_theta_i = v_theta_n - (Jr * Bz) / (ni * mi * nu_in)
    
    Parameters:
    -----------
    vtheta_out : 2D array (Nr, Nz) to be updated in-place
    Jr         : 2D array (Nr, Nz), Total radial Current Density
    Bz         : 2D array (Nr, Nz), Axial Magnetic Field
    ni         : 2D array (Nr, Nz), Ion Density
    nu_in      : 2D array (Nr, Nz), Ion-Neutral Collision Freq
    un_theta   : 2D array (Nr, Nz), Neutral Gas Velocity
    mask       : 2D array (Nr, Nz), 1=Plasma, 0=Solid
    mi         : float, Ion Mass
    """
    Nr, Nz = Jr.shape
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                # Local scalar values
                n_local = ni[i, j]
                nu_local = nu_in[i, j]
                                    
                # Force calculation
                # Lorentz Force term (assuming J_r x B_z -> -theta direction)
                F_lorentz = -1.0 * Jr[i, j] * Bz[i, j]
                    
                # Drag coefficient = rho_i * nu_in
                drag_coeff = mi * n_local * nu_local
                    
                # Algebraic Solution
                vtheta_out[i, j] = un_theta[i, j] + (F_lorentz / drag_coeff)
                    
            else:
                # Solid boundaries
                vtheta_out[i, j] = 0.0

@njit(cache=True)
def compute_nu_i_kernel(nu_i_out, ni, Ti, nn, Tn, Z, mi, sigma_cx, mask):
    """
    Computes total ion collision frequency: nu_i = nu_ii + nu_in
    Currently only includes nu_in via charge exchange.
    nu_in (Charge Exchange): nn * sigma_cx * v_thermal_rel
    """
    Nr, Nz = nu_i_out.shape
    
    # Physical Constants
    kb = constants.kb    

    # --- Pre-calculate Constants for nu_in ---
    # Relative thermal velocity factor: sqrt( 8*kB / (pi*mi) ) * sqrt(Ti + Tn)
    # Assuming mi approx mn
    factor_in = np.sqrt(8.0 * kb / (np.pi * mi))

    min_T = 300.0 # Avoid division by zero temperature

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

                    # 2. Charge Exchange (nu_in)
                    v_rel = factor_in * np.sqrt(Ti_val + Tn_val)
                    nu_in = nn_val * sigma_cx * v_rel

                    nu_i_out[i, j] = nu_in
                else:
                    nu_i_out[i, j] = 0.0
            else:
                nu_i_out[i, j] = 0.0

@njit(cache=True)
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

@njit(cache=True)
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

@njit(cache=True)
def update_Ti_joule_heating_kernel(Ti_out, Tn, Te, 
                                   Q_Joule_ions,
                                   ni, nu_in, nu_ei, mi, mn, mask, me, kb):
    """
    Updates Ion Temperature (Ti) using explicit Joule Heating (J*E) as the source.
    
    Balance:
    (J_perp^2 / sigma_P) + (J_par^2 / sigma_par) + Q_ie = Q_in_thermal
    
    where Q_in_thermal = 3 * (mi/mn) * ni * nu_in * kb * (Ti - Tn)
    """
    Nr, Nz = Ti_out.shape
    
    ratio_me_mi = me / mi
    ratio_mi_mn = mi / mn

    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n_local = ni[i, j]
                
                if n_local > 1e10:
                    # --- 1. Joule Heating (Source) ---
                    Q_joule = Q_Joule_ions[i, j]
                   
                    # --- 2. Electron-Ion Heat Transfer (Source/Sink) ---
                    Te_local = Te[i, j]
                    Tn_local = Tn[i, j]
                    
                    # nu_ei
                    nu_ei_local = nu_ei[i, j]

                    # Q_ie coeff: A * (Te - Ti)
                    # A = 3 * (me/mi) * n * nu_ei * kb
                    A_coeff = 3.0 * ratio_me_mi * n_local * nu_ei_local * kb
                    
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

@njit(cache=True)
def solve_vtheta_viscous_SOR(vtheta, Jr, Bz, ni, nu_in, un_theta, eta, 
                             mask, dr, dz, r_coords, mi, 
                             max_iter=2000, tol=1e-5, omega=1.4):
    """
    Solves steady-state viscous momentum equation using SOR (Successive Over-Relaxation).
    
    Equation: 
      0 = F_lorentz - Drag + Viscosity
      F_lorentz = -Jr * Bz
      Drag = rho * nu * (v_i - v_n)
      Viscous = eta * [ d2v/dr2 + (1/r)dv/dr - v/r^2 + d2v/dz2 ] 
                (Simplified Laplacian form for cylindrical vector)
    
    Boundary Conditions:
      - Mask=0 (Walls/Internal): v = 0
      - r=0 (Axis): v = 0
      - z=0 (Inlet): v = 0
      - z=L (Outlet): dv/dz = 0 (v_last = v_second_last)
    """
    Nr, Nz = vtheta.shape
    
    # 1. Pre-Clean Solids
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0:
                vtheta[i, j] = 0.0
            if i == 0 or i == Nr - 1 or j == 0: # Domain boundaries
                vtheta[i, j] = 0.0

    # Geometric factors
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)
    inv_2dr = 1.0 / (2.0 * dr)
    
    for k in range(max_iter):
        max_diff = 0.0
        
        for i in range(1, Nr - 1): 
            r_local = r_coords[i]
            inv_r = 1.0 / r_local
            inv_r2 = inv_r * inv_r
            
            for j in range(1, Nz - 1): 
                
                # Check 1: Is this plasma?
                if mask[i, j] == 1:
                    
                    # Check 2: Is it TOUCHING a wall? (The "Padding" Fix)
                    # If any neighbor is 0 (Solid), force this node to 0 (No-Slip Wall)
                    if (mask[i+1, j] == 0 or mask[i-1, j] == 0 or 
                        mask[i, j+1] == 0 or mask[i, j-1] == 0):
                        
                        vtheta[i, j] = 0.0
                        continue # Skip to next node
                    
                    # --- Standard Physics Solver for Bulk Plasma ---
                    
                    n_loc = ni[i, j]
                    nu_loc = nu_in[i, j]
                    eta_loc = eta[i, j]
                    
                    # Physics Terms
                    C_drag = mi * n_loc * nu_loc
                    F_L = -1.0 * Jr[i, j] * Bz[i, j]
                    
                    # Neighbors
                    v_ip = vtheta[i+1, j]
                    v_im = vtheta[i-1, j]
                    v_jp = vtheta[i, j+1]
                    v_jm = vtheta[i, j-1]
                    
                    # Discretization
                    visc_r_part = eta_loc * ( (v_ip + v_im) * inv_dr2 + (v_ip - v_im) * inv_2dr * inv_r )
                    visc_z_part = eta_loc * ( (v_jp + v_jm) * inv_dz2 )
                    
                    RHS = F_L + (C_drag * un_theta[i, j]) + visc_r_part + visc_z_part
                    Coeff = C_drag + eta_loc * (2.0*inv_dr2 + 2.0*inv_dz2 + inv_r2)
                    
                    if Coeff > 1e-20:
                        v_star = RHS / Coeff
                        diff = abs(v_star - vtheta[i, j])
                        vtheta[i, j] = (1.0 - omega) * vtheta[i, j] + omega * v_star
                        if diff > max_diff:
                            max_diff = diff
                            
                else:
                    # Solid Node
                    vtheta[i, j] = 0.0
        
        # --- Boundaries (Neumann Z-Max) ---
        for i in range(Nr):
            if mask[i, Nz-1] == 1:
                # Only copy if NOT touching a side wall
                if mask[i+1, Nz-1] == 1 and mask[i-1, Nz-1] == 1:
                    vtheta[i, Nz-1] = vtheta[i, Nz-2]
                else:
                    vtheta[i, Nz-1] = 0.0
        
        if max_diff < tol:
            break