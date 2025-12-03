import numpy as np
from numba import njit

from centrifugesim import constants

MU0 = constants.mu0

@njit
def _sor_step(A, r, dr, dz, sigma, omega, Jcoil, w_relax, mask):
    """
    SOR sweep with support for internal metal structures (mask=0).
    mask: 1=Plasma/Vacuum (Solve), 0=Metal (A=0).
    """
    Nr, Nz = A.shape
    inv_dr2 = 1.0/(dr*dr)
    inv_dz2 = 1.0/(dz*dz)
    
    # Interior Points
    for i in range(1, Nr-1):
        # Optimization: Check if entire column is vacuum/plasma or mixed? 
        # For now, check point-by-point which is robust.
        
        ri = r[i]
        inv_r = 1.0/ri
        half_inv_r_dr = 0.5*inv_r/dr 

        for j in range(1, Nz-1):
            # --- METAL CHECK ---
            if mask[i, j] == 0:
                A[i, j] = 0.0 + 0.0j
                continue
            # -------------------

            # Standard Stencil
            c_ip = inv_dr2 + half_inv_r_dr
            c_im = inv_dr2 - half_inv_r_dr
            c_jp = inv_dz2
            c_jm = inv_dz2
            c0   = -(2.0*inv_dr2 + 2.0*inv_dz2) - (1.0/(ri*ri)) + 1j*MU0*omega*sigma[i,j]

            rhs = -MU0*Jcoil[i,j]
            neigh = c_ip*A[i+1,j] + c_im*A[i-1,j] + c_jp*A[i,j+1] + c_jm*A[i,j-1]
            A_new = (rhs - neigh) / c0

            A[i,j] = (1.0 - w_relax)*A[i,j] + w_relax*A_new

    # --- Boundaries ---

    # 1. Axis r=0: Dirichlet A=0 (Geometry)
    for j in range(Nz):
        A[0, j] = 0.0 + 0.0j

    # 2. Outer "Far Field" r=rmax_extended: Open/Neumann
    # Note: If mask is 0 at the edge, this will be overwritten, but usually 
    # far field is vacuum (mask=1).
    for j in range(Nz):
        A[Nr-1, j] = A[Nr-2, j]

    # 3. Bottom z=zmin (j=0): METAL WALL (Dirichlet)
    # User specified: "nodes (:,0) ... there is a metallic wall"
    for i in range(Nr):
        A[i, 0] = 0.0 + 0.0j

    # 4. Top z=zmax (j=Nz-1): Neumann (Symmetry/Open)
    # (Unless user specifies top cap is also metal, we keep this open/gradient=0)
    for i in range(Nr):
        A[i, Nz-1] = A[i, Nz-2]


@njit
def _compute_residual_inf(A, r, dr, dz, sigma, omega, Jcoil, mask):
    """Residual check ignoring metal points."""
    Nr, Nz = A.shape
    inv_dr2 = 1.0/(dr*dr)
    inv_dz2 = 1.0/(dz*dz)
    res_inf = 0.0
    
    for i in range(1, Nr-1):
        ri = r[i]
        inv_r = 1.0/ri
        half_inv_r_dr = 0.5*inv_r/dr
        for j in range(1, Nz-1):
            
            # If metal, residual is effectively 0 (constraint satisfied)
            if mask[i, j] == 0:
                continue

            lap_r = inv_dr2*(A[i+1,j] - 2.0*A[i,j] + A[i-1,j]) + half_inv_r_dr*(A[i+1,j] - A[i-1,j])
            lap_z = inv_dz2*(A[i,j+1] - 2.0*A[i,j] + A[i,j-1])
            L_A  = lap_r + lap_z - (A[i,j]/(ri*ri)) + 1j*MU0*omega*sigma[i,j]*A[i,j]
            S    = -MU0*Jcoil[i,j]
            res  = L_A - S
            
            mag = (res.real*res.real + res.imag*res.imag)**0.5
            if mag > res_inf:
                res_inf = mag
    return res_inf


def solve_Atheta_sor(r, z, sigma, omega, Jcoil, mask, w_relax=1.6, tol=1e-8, maxiter=20000, A0=None, verbose=True):
    """
    Main solver driver. Now accepts 'mask'.
    """
    r = np.asarray(r)
    z = np.asarray(z)
    dr = float(r[1]-r[0])
    dz = float(z[1]-z[0])
    Nr, Nz = sigma.shape
    
    A = np.zeros((Nr, Nz), dtype=np.complex128) if A0 is None else A0.astype(np.complex128).copy()
    
    # Initial residual
    res0 = _compute_residual_inf(A, r, dr, dz, sigma, omega, Jcoil, mask)
    if res0 == 0.0: res0 = 1.0
    res_hist = [res0]

    for it in range(1, maxiter+1):
        _sor_step(A, r, dr, dz, sigma, omega, Jcoil, w_relax, mask)
        
        # Check convergence less frequently to save time
        if it % 100 == 0:
            res = _compute_residual_inf(A, r, dr, dz, sigma, omega, Jcoil, mask)
            res_hist.append(res)
            if verbose and (it % 1000 == 0 or it < 500):
                print(f"Iter {it:6d}: residual = {res:.3e} (rel {res/res0:.3e})")
            
            if res <= tol*res0:
                if verbose: print(f"Converged at iter {it}")
                return A, it, res_hist

    if verbose:
        print(f"[WARN] Reached maxiter={maxiter} with residual {res:.3e}")
    return A, it, res_hist


def solve_with_vacuum_extension(geom, sigma_plasma, omega, coil_config, w_relax=1.6, tol=1e-8, maxiter=50000, A0=None, verbose=True):
    """
    Extended grid wrapper.
    Handles:
    1. Grid extension.
    2. Mapping geom.mask (internal metal).
    3. Applying 'Anode' wall at r=rmax boundary.
    """
    
    # --- 1. Grid Extension ---
    r_plasma_max = geom.r[-1]
    r_coil_max = coil_config['r2']
    
    if r_coil_max > r_plasma_max:
        # Extend grid by coil outer edge + buffer
        buffer_dist = r_coil_max - r_plasma_max
        r_target_max = r_coil_max + buffer_dist
        new_r_points = np.arange(r_plasma_max + geom.dr, r_target_max, geom.dr)
        r_extended = np.concatenate([geom.r, new_r_points])
        if verbose:
            print(f"Grid extended: Nr {geom.Nr} -> {len(r_extended)}")
    else:
        r_extended = geom.r
        if verbose: print("No grid extension needed.")

    Nr_ext = len(r_extended)
    Nz = geom.Nz
    
    # --- 2. Setup Extended Physics Arrays ---
    
    # Sigma: Pad with zeros (Vacuum)
    sigma_extended = np.zeros((Nr_ext, Nz), dtype=sigma_plasma.dtype)
    sigma_extended[:geom.Nr, :] = sigma_plasma

    # Mask: Pad with 1s (Vacuum)
    # (Vacuum allows fields to exist, so mask=1)
    mask_extended = np.ones((Nr_ext, Nz), dtype=np.int32)
    
    # Copy original internal mask (handles internal electrodes/metal)
    if hasattr(geom, 'mask'):
        mask_extended[:geom.Nr, :] = geom.mask
    else:
        # Fallback if geom has no mask, assume all plasma region is valid
        pass 

    # --- 3. Apply Wall Conditions (Anode) ---
    # User Req: "nodes that correspond to geom.rmax ... and z from geom.zmin_anode to geom.zmax ... correspond to metallic wall"
    # Logic: Finds the index corresponding to original rmax (geom.Nr - 1)
    # and sets mask to 0 for the specified Z range.
    
    idx_rmax = geom.Nr - 1
    
    # Ensure zmin_anode exists in geom
    zmin_anode = getattr(geom, 'zmin_anode', 0.0) # Default to 0 if not set, or raise error?
    
    # Find Z indices
    # We want z >= zmin_anode. 
    z_indices = np.where((geom.z >= zmin_anode) & (geom.z <= geom.z[-1]))[0]
    
    # Paint the wall onto the mask
    mask_extended[idx_rmax, z_indices] = 0

    # --- 4. Coil Current ---
    J_extended = np.zeros((Nr_ext, Nz), dtype=np.complex128)
    
    if 'J0' in coil_config:
        J_val = coil_config['J0']
    elif 'I_total' in coil_config:
        area = (coil_config['r2'] - coil_config['r1']) * (coil_config['z2'] - coil_config['z1'])
        J_val = coil_config['I_total'] / area
    else:
        raise ValueError("coil_config must have 'J0' or 'I_total'")

    r_mask = (r_extended >= coil_config['r1']) & (r_extended <= coil_config['r2'])
    z_mask = (geom.z >= coil_config['z1']) & (geom.z <= coil_config['z2'])
    J_extended[np.ix_(r_mask, z_mask)] = J_val

    # --- 5. Solve ---
    A_full, iters, _ = solve_Atheta_sor(
        r=r_extended,
        z=geom.z,
        sigma=sigma_extended,
        omega=omega,
        Jcoil=J_extended,
        mask=mask_extended,  # Pass the new mask
        w_relax=w_relax,
        tol=tol,
        maxiter=maxiter,
        A0=A0,
        verbose=verbose
    )

    # --- 6. Slice and Return ---
    return A_full, A_full[:geom.Nr, :]


def calculate_deposited_power(A_plasma, sigma, omega, geom):
    """
    Calculates the Ohmic heating power deposited into the plasma.
    
    Parameters
    ----------
    A_plasma : (Nr, Nz) complex array
        The vector potential solution (sliced to plasma region).
    sigma    : (Nr, Nz) float array
        The Pedersen conductivity [S/m].
    omega    : float
        Angular frequency [rad/s].
    geom     : object
        Geometry object with attributes .r (array), .dr, .dz.
        
    Returns
    -------
    P_density : (Nr, Nz) float array
        Power density field [Watts/m^3].
    P_total   : float
        Total power absorbed by the plasma volume [Watts].
    """
    
    # 1. Calculate Electric Field Amplitude (E = -i * w * A)
    # We only need the magnitude for power: |E| = w * |A|
    E_mag = omega * np.abs(A_plasma)
    
    # 2. Calculate Power Density [W/m^3]
    # Formula: 0.5 * sigma * |E|^2
    # The 0.5 comes from time-averaging sinusoidal fields (RMS factor)
    P_density = 0.5 * sigma * (E_mag**2)
    
    # 3. Integrate to find Total Power [Watts]
    # Volume element dV = 2 * pi * r * dr * dz
    # We assume cylindrical symmetry
    
    # Create 2D r array for broadcasting
    r_2d = geom.r[:, np.newaxis] 
    
    # Integration summation
    # sum( P_dens * 2*pi*r ) * dr * dz
    integral_term = P_density * (2.0 * np.pi * r_2d)
    P_total = np.sum(integral_term) * geom.dr * geom.dz
    
    return P_density, P_total