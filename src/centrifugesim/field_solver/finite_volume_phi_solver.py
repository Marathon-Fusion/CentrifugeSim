import numpy as np
from numba import njit, prange

from centrifugesim import constants


@njit(parallel=True, fastmath=True, cache=True)
def _compute_EJ_core(phi, mask, r, z,
                     sigma_P, sigma_par,
                     ne, pe, Bz, un_theta, Ji_r, Ji_z,
                     ne_floor, echarge,
                     use_pe, use_rot, use_ji,
                     fill_solid_with_nan):
    """
    Compute E and J with mask-aware one-sided (at boundaries/solids)
    and 3-point nonuniform centered derivatives in the interior.
    """
    Nr, Nz = phi.shape
    Er  = np.zeros((Nr, Nz), dtype=phi.dtype)
    Ez  = np.zeros((Nr, Nz), dtype=phi.dtype)
    Jr  = np.zeros((Nr, Nz), dtype=phi.dtype)
    Jz  = np.zeros((Nr, Nz), dtype=phi.dtype)

    EPS = 1e-30

    for i in prange(Nr):
        # local spacings in r
        if i > 0:
            drL = r[i] - r[i-1]
        else:
            drL = 1.0
        if i < Nr-1:
            drR = r[i+1] - r[i]
        else:
            drR = 1.0

        for j in range(Nz):
            # skip solids; optionally fill with NaN
            if mask[i, j] == 0:
                if fill_solid_with_nan:
                    Er[i, j] = np.nan
                    Ez[i, j] = np.nan
                    Jr[i, j] = np.nan
                    Jz[i, j] = np.nan
                else:
                    Er[i, j] = 0.0
                    Ez[i, j] = 0.0
                    Jr[i, j] = 0.0
                    Jz[i, j] = 0.0
                continue

            # local spacings in z
            if j > 0:
                dzS = z[j] - z[j-1]
            else:
                dzS = 1.0
            if j < Nz-1:
                dzN = z[j+1] - z[j]
            else:
                dzN = 1.0

            # neighbor availability (stay within plasma only)
            left_ok   = (i > 0)    and (mask[i-1, j] == 1)
            right_ok  = (i < Nr-1) and (mask[i+1, j] == 1)
            south_ok  = (j > 0)    and (mask[i, j-1] == 1)
            north_ok  = (j < Nz-1) and (mask[i, j+1] == 1)

            # ---------------------------
            # dphi/dr (mask-aware)
            # ---------------------------
            if left_ok and right_ok:
                # 3-point nonuniform centered derivative via Lagrange weights
                hL = max(drL, EPS)
                hR = max(drR, EPS)
                denom = hL + hR
                w_im1 = -hR / (hL * denom)
                w_i   =  (hR - hL) / (hL * hR)
                w_ip1 =  hL / (hR * denom)
                dphidr = w_im1*phi[i-1, j] + w_i*phi[i, j] + w_ip1*phi[i+1, j]
            elif right_ok:
                # forward one-sided
                h = max(drR, EPS)
                dphidr = (phi[i+1, j] - phi[i, j]) / h
            elif left_ok:
                # backward one-sided
                h = max(drL, EPS)
                dphidr = (phi[i, j] - phi[i-1, j]) / h
            else:
                dphidr = 0.0  # isolated plasma pixel

            # ---------------------------
            # dphi/dz (mask-aware)
            # ---------------------------
            if south_ok and north_ok:
                hS = max(dzS, EPS)
                hN = max(dzN, EPS)
                denom = hS + hN
                w_jm1 = -hN / (hS * denom)
                w_j   =  (hN - hS) / (hS * hN)
                w_jp1 =  hS / (hN * denom)
                dphidz = w_jm1*phi[i, j-1] + w_j*phi[i, j] + w_jp1*phi[i, j+1]
            elif north_ok:
                h = max(dzN, EPS)
                dphidz = (phi[i, j+1] - phi[i, j]) / h
            elif south_ok:
                h = max(dzS, EPS)
                dphidz = (phi[i, j] - phi[i, j-1]) / h
            else:
                dphidz = 0.0

            # Electric field (E = -grad phi)
            Er_ij = -dphidr
            Ez_ij = -dphidz
            Er[i, j] = Er_ij
            Ez[i, j] = Ez_ij

            # Base Ohmic currents
            sigP   = sigma_P[i, j]
            sigPar = sigma_par[i, j]
            jr = sigP   * Er_ij
            jz = sigPar * Ez_ij

            # Optional pressure-battery and rotation contributions
            if use_pe:
                # dp/dr
                if left_ok and right_ok:
                    hL = max(drL, EPS)
                    hR = max(drR, EPS)
                    denom = hL + hR
                    w_im1 = -hR / (hL * denom)
                    w_i   =  (hR - hL) / (hL * hR)
                    w_ip1 =  hL / (hR * denom)
                    dpdr = w_im1*pe[i-1, j] + w_i*pe[i, j] + w_ip1*pe[i+1, j]
                elif right_ok:
                    h = max(drR, EPS)
                    dpdr = (pe[i+1, j] - pe[i, j]) / h
                elif left_ok:
                    h = max(drL, EPS)
                    dpdr = (pe[i, j] - pe[i-1, j]) / h
                else:
                    dpdr = 0.0

                # dp/dz
                if south_ok and north_ok:
                    hS = max(dzS, EPS)
                    hN = max(dzN, EPS)
                    denom = hS + hN
                    w_jm1 = -hN / (hS * denom)
                    w_j   =  (hN - hS) / (hS * hN)
                    w_jp1 =  hS / (hN * denom)
                    dpdz = w_jm1*pe[i, j-1] + w_j*pe[i, j] + w_jp1*pe[i, j+1]
                elif north_ok:
                    h = max(dzN, EPS)
                    dpdz = (pe[i, j+1] - pe[i, j]) / h
                elif south_ok:
                    h = max(dzS, EPS)
                    dpdz = (pe[i, j] - pe[i, j-1]) / h
                else:
                    dpdz = 0.0

                n_e = ne[i, j]
                if n_e < ne_floor:
                    n_e = ne_floor
                inv_e_ne = 1.0 / (echarge * n_e)
                jr += sigP   * (inv_e_ne * dpdr)
                jz += sigPar * (inv_e_ne * dpdz)

            if use_rot:
                jr += sigP * (Bz[i, j] * un_theta[i, j])

            if use_ji:
                jr += Ji_r[i, j]
                jz += Ji_z[i, j]

            Jr[i, j] = jr
            Jz[i, j] = jz

    return Er, Ez, Jr, Jz


def compute_E_and_J(phi, geom,
                    sigma_P, sigma_parallel,
                    ne=None, pe=None,
                    Bz=None, un_theta=None,
                    Ji_r=None, Ji_z=None,
                    ne_floor=1.0,
                    fill_solid_with_nan=False):
    """
    Compute E=(Er, Ez) and J=(Jr, Jz) on a masked RZ grid using
    one-sided derivatives at boundaries/solids and 3-point nonuniform
    centered derivatives in the interior.
    """
    r = np.asarray(geom.r, dtype=np.float64)
    z = np.asarray(geom.z, dtype=np.float64)
    mask = np.asarray(geom.mask, dtype=np.int8)

    phi = np.asarray(phi, dtype=np.float64)
    sigma_P   = np.asarray(sigma_P,   dtype=np.float64)
    sigma_par = np.asarray(sigma_parallel, dtype=np.float64)

    use_pe  = (ne is not None) and (pe is not None)
    use_rot = (Bz is not None) and (un_theta is not None)
    use_ji  = (Ji_r is not None) or (Ji_z is not None)

    Nr, Nz = phi.shape
    zero = np.zeros_like(phi)

    ne_arr       = np.asarray(ne,       dtype=np.float64) if use_pe  else zero
    pe_arr       = np.asarray(pe,       dtype=np.float64) if use_pe  else zero
    Bz_arr       = np.asarray(Bz,       dtype=np.float64) if use_rot else zero
    utheta_arr   = np.asarray(un_theta, dtype=np.float64) if use_rot else zero
    Ji_r_arr     = np.asarray(Ji_r,     dtype=np.float64) if (Ji_r is not None) else zero
    Ji_z_arr     = np.asarray(Ji_z,     dtype=np.float64) if (Ji_z is not None) else zero

    echarge = constants.q_e

    Er, Ez, Jr, Jz = _compute_EJ_core(phi, mask, r, z,
                                      sigma_P, sigma_par,
                                      ne_arr, pe_arr, Bz_arr, utheta_arr, Ji_r_arr, Ji_z_arr,
                                      float(ne_floor), float(echarge),
                                      use_pe, use_rot, use_ji,
                                      fill_solid_with_nan)
    return Er, Ez, Jr, Jz


@njit(parallel=True, fastmath=True, cache=True)
def _assemble_coefficients_core(
    Nr, Nz,
    r_c, r_w, r_e, dr_w, dr_e, dr_i,
    dz_s, dz_n, dz_j,
    sigP_e, sigPar_n,                 # face conductivities
    sigmaP_cell, sigmaPar_cell,       # cell-centered conductivities
    mask_u8, cathode_u8, anode_u8,    # uint8 masks (1=True, 0=False)
    rmax_dirichlet_u8,                # uint8 of length Nz (1=Dirichlet at rmax for this j)
    g_top_cathode,                    # length Nr: dphi/dz at top-of-cathode, Neumann array
    phi_dirichlet_val,                # typically 0.0 (anode potential)
    S                                 # source term on cell centers
    ):
    aE = np.zeros((Nr, Nz))
    aW = np.zeros((Nr, Nz))
    aN = np.zeros((Nr, Nz))
    aS = np.zeros((Nr, Nz))
    aP = np.zeros((Nr, Nz))
    b  = np.zeros((Nr, Nz))

    for j in prange(Nz):
        for i in range(Nr):
            # Skip solid cells entirely (fix them to phi = phi_dirichlet_val so they don't participate)
            if mask_u8[i, j] == 0:
                aP[i, j] = 1.0
                aE[i, j] = 0.0
                aW[i, j] = 0.0
                aN[i, j] = 0.0
                aS[i, j] = 0.0
                b[i, j]  = phi_dirichlet_val  # sets phi=0 for solids (anodes/cathode)
                continue

            # --- EAST face (radial +) ---
            if i < Nr - 1:
                if mask_u8[i+1, j] == 1:
                    # plasma-plasma interior face
                    ke = sigP_e[i, j]
                    ae = (r_e[i] * ke) / (dr_e[i] * dr_i[i])
                    aE[i, j] = ae
                else:
                    # plasma-solid face
                    if anode_u8[i+1, j] == 1:
                        # Dirichlet φ = phi_dirichlet_val (anode)
                        ke = sigmaP_cell[i, j]  # use cell conductivity for boundary face
                        ae_face = (r_e[i] * ke) / (dr_e[i] * dr_i[i])
                        aP[i, j] += ae_face
                        b[i, j]  += ae_face * phi_dirichlet_val
                        aE[i, j]  = 0.0
                    else:
                        # cathode (or other solid): Neumann zero flux on radial face
                        aE[i, j] = 0.0
            else:
                # r = rmax boundary
                if rmax_dirichlet_u8[j] == 1:
                    # Dirichlet on outer wall
                    if Nr > 1:
                        ke_edge = sigmaP_cell[i, j]  # use cell conductivity
                        ae_face = (r_e[i] * ke_edge) / (dr_e[i] * dr_i[i])
                        aP[i, j] += ae_face
                        b[i, j]  += ae_face * phi_dirichlet_val
                # else Neumann zero radial flux
                aE[i, j] = 0.0

            # --- WEST face (radial -) ---
            if i > 0:
                if mask_u8[i-1, j] == 1:
                    kw = sigP_e[i-1, j]
                    aw = (r_w[i] * kw) / (dr_w[i] * dr_i[i])
                    aW[i, j] = aw
                else:
                    if anode_u8[i-1, j] == 1:
                        # Dirichlet on inner anode face (r = rmin_anode)
                        kw = sigmaP_cell[i, j]
                        aw_face = (r_w[i] * kw) / (dr_w[i] * dr_i[i])
                        aP[i, j] += aw_face
                        b[i, j]  += aw_face * phi_dirichlet_val
                        aW[i, j]  = 0.0
                    else:
                        # cathode vertical face: Neumann zero
                        aW[i, j] = 0.0
            else:
                # axis r=0 symmetry: Neumann zero (already aW=0)
                aW[i, j] = 0.0

            # --- NORTH face (axial +) ---
            if j < Nz - 1:
                if mask_u8[i, j+1] == 1:
                    kn = sigPar_n[i, j]
                    an = (r_c[i] * kn) / (dz_n[j] * dz_j[j])
                    aN[i, j] = an
                else:
                    if anode_u8[i, j+1] == 1:
                        # Dirichlet on anode axial face
                        kn = sigmaPar_cell[i, j]
                        an_face = (r_c[i] * kn) / (dz_n[j] * dz_j[j])
                        aP[i, j] += an_face
                        b[i, j]  += an_face * phi_dirichlet_val
                        aN[i, j]  = 0.0
                    else:
                        # solid (unlikely north-of-plasma cathode), Neumann zero
                        aN[i, j] = 0.0
            else:
                # z = zmax: Neumann zero
                aN[i, j] = 0.0

            # --- SOUTH face (axial -) ---
            if j > 0:
                if mask_u8[i, j-1] == 1:
                    ks = sigPar_n[i, j-1]
                    as_ = (r_c[i] * ks) / (dz_s[j] * dz_j[j])
                    aS[i, j] = as_
                else:
                    if anode_u8[i, j-1] == 1:
                        # Dirichlet on anode axial face
                        ks = sigmaPar_cell[i, j]
                        as_face = (r_c[i] * ks) / (dz_s[j] * dz_j[j])
                        aP[i, j] += as_face
                        b[i, j]  += as_face * phi_dirichlet_val
                        aS[i, j]  = 0.0
                    elif cathode_u8[i, j-1] == 1:
                        # Top of cathode: impose ∂φ/∂z = g_top_cathode[i]
                        #ks = sigPar_n[i, j-1] if Nz > 1 else 0.0
                        ks = sigPar_n[i, j]
                        qzS = ks * g_top_cathode[i]
                        b[i, j] += r_c[i] * qzS / dz_j[j]
                        aS[i, j]  = 0.0
                    else:
                        # other solid below: Neumann zero
                        aS[i, j] = 0.0
            else:
                # z = zmin (bottom): Neumann zero everywhere there is plasma (outside cathode)
                aS[i, j] = 0.0

            # Final diagonal and source
            aP[i, j] += (aE[i, j] + aW[i, j] + aN[i, j] + aS[i, j])
            b[i, j]  += r_c[i] * S[i, j]

    return aP, aE, aW, aN, aS, b


def build_geometry_faces(r, z):
    r = np.asarray(r)
    z = np.asarray(z)
    Nr = r.size
    Nz = z.size

    # radial faces and widths
    r_c = r.copy()
    r_e = np.empty_like(r_c)
    r_w = np.empty_like(r_c)
    dr_e = np.empty_like(r_c)
    dr_w = np.empty_like(r_c)

    r_e[:-1] = 0.5*(r[:-1] + r[1:])
    r_e[-1]  = r[-1] + 0.5*(r[-1] - r[-2]) if Nr > 1 else r[-1]
    dr_e[:-1] = r[1:] - r[:-1]
    dr_e[-1]  = (r[-1] - r[-2]) if Nr > 1 else 1.0

    r_w[0]    = 0.0
    r_w[1:]   = 0.5*(r[:-1] + r[1:])
    dr_w[0]   = (r[1] - r[0]) if Nr > 1 else 1.0
    dr_w[1:]  = r[1:] - r[:-1]

    dr_i = r_e - r_w  # control-volume radial width

    # axial faces and widths
    z_c = z.copy()
    z_n = np.empty_like(z_c)
    z_s = np.empty_like(z_c)
    dz_n = np.empty_like(z_c)
    dz_s = np.empty_like(z_c)

    z_n[:-1] = 0.5*(z[:-1] + z[1:])
    z_n[-1]  = z[-1] + 0.5*(z[-1] - z[-2]) if Nz > 1 else z[-1]
    dz_n[:-1] = z[1:] - z[:-1]
    dz_n[-1]  = (z[-1] - z[-2]) if Nz > 1 else 1.0

    z_s[0]   = z[0]
    z_s[1:]  = 0.5*(z[:-1] + z[1:])
    dz_s[0]  = 0.5*(z[1] - z[0]) if Nz > 1 else 1.0
    dz_s[1:] = z[1:] - z[:-1]

    dz_j = z_n - z_s  # control-volume axial width

    return (r_c, r_w, r_e, dr_w, dr_e, dr_i,
            z_c, z_s, z_n, dz_s, dz_n, dz_j)


def build_face_conductivities(sigmaP, sigmaPar, eps=1e-30):
    # radial faces (east): shape (Nr-1, Nz)
    sigP_e = (2.0 * sigmaP[:-1, :] * sigmaP[1:, :]) / (sigmaP[:-1, :] + sigmaP[1:, :] + eps)

    # axial faces (north): shape (Nr, Nz-1)
    sigPar_n = (2.0 * sigmaPar[:, :-1] * sigmaPar[:, 1:]) / (sigmaPar[:, :-1] + sigmaPar[:, 1:] + eps)

    return sigP_e, sigPar_n


def assemble_coefficients(
    r, z, sigmaP, sigmaPar,
    geom,
    dphi_dz_cathode_top=None,         # shape (Nr,), gradient at z=zmax_cathode (on plasma side)
    phi_anode_value=0.0,
    S=None
):
    """
    Assemble FV coefficients for:
        (1/r) ∂_r [ r σ_P ∂_r φ ] + ∂_z [ σ_|| ∂_z φ ] = -S

    Boundary conditions (from geom):
      - Axis r=0:                         Neumann (∂φ/∂r = 0)
      - z=0 outside cathode footprint:    Neumann (∂φ/∂z = 0)
      - z=zmax:                           Neumann (∂φ/∂z = 0)
      - Cathode vertical face:            Neumann (∂φ/∂r = 0)
      - Cathode top (z=zmax_cathode):     Neumann array ∂φ/∂z = dphi_dz_cathode_top[i]
      - Any plasma↔anode face:            Dirichlet φ = phi_anode_value
      - r=rmax:  Neumann for z < zmin_anode, Dirichlet (φ=phi_anode_value) for z ≥ zmin_anode
    """
    r = np.asarray(r); z = np.asarray(z)
    Nr = r.size; Nz = z.size

    (r_c, r_w, r_e, dr_w, dr_e, dr_i,
     z_c, z_s, z_n, dz_s, dz_n, dz_j) = build_geometry_faces(r, z)

    sigP_e, sigPar_n = build_face_conductivities(sigmaP, sigmaPar)

    # Masks: convert to uint8 for numba
    mask_u8    = geom.mask.astype(np.uint8)
    cathode_u8 = geom.cathode_mask.astype(np.uint8)
    anode_u8   = (geom.anode1_mask | geom.anode2_mask).astype(np.uint8)

    # Outer wall r=rmax: Dirichlet φ=0 for z >= zmin_anode, else Neumann zero flux
    rmax_dirichlet_by_j = (z >= geom.zmin_anode).astype(np.uint8)

    # Cathode top Neumann array
    if dphi_dz_cathode_top is None:
        g_top = np.zeros(Nr, dtype=sigmaP.dtype)
    else:
        g_top = np.asarray(dphi_dz_cathode_top, dtype=sigmaP.dtype)
        if g_top.shape != (Nr,):
            raise ValueError("dphi_dz_cathode_top must be shape (Nr,)")

    # Source term
    if S is None:
        S = np.zeros((Nr, Nz), dtype=sigmaP.dtype)
    # Do not source inside solids
    S = np.where(mask_u8 == 1, S, 0.0)

    aP, aE, aW, aN, aS, b = _assemble_coefficients_core(
        Nr, Nz,
        r_c, r_w, r_e, dr_w, dr_e, dr_i,
        dz_s, dz_n, dz_j,
        sigP_e, sigPar_n,
        sigmaP, sigmaPar,
        mask_u8, cathode_u8, anode_u8,
        rmax_dirichlet_by_j,
        g_top, phi_anode_value, S
    )
    return aP, aE, aW, aN, aS, b


def compute_source_S(r, z,
                     sigma_P, sigma_parallel,
                     ne, pe,
                     ne_floor,
                     Bz=None, un_theta=None,
                     Ji_r=None, Ji_z=None,
                     mask=None,
                     eps=1e-30):
    """
    Build S(r,z) for:
        -(1/r) ∂_r [ r σ_P ∂_r φ ] - ∂_z [ σ_|| ∂_z φ ] = S

    where
        -S = (1/r) ∂_r { r σ_P [ (1/e ne) ∂_r p_e + Bz u_{nθ} ] }
            + ∂_z { σ_|| [ (1/e ne) ∂_z p_e ] }
            + (1/r) ∂_r ( r J_{i,r} ) + ∂_z J_{i,z}

    Notes
    -----
    * Uses the *plus* sign in front of the ion-current divergence terms.
    * ne, pe, sigma_P, sigma_parallel are cell-centered (shape Nr x Nz).
    * Ji_r and Ji_z are cell-centered; they are averaged to faces internally.
    * Bz and un_theta are optional; if omitted, that term is treated as zero.
    * Returns S on cell centers (Nr x Nz), ready to pass into assemble_coefficients.
    """
    Nr, Nz = sigma_P.shape
    echarge = constants.q_e

    # --- geometry and face conductivities (consistent with your assembly) ---
    (r_c, r_w, r_e, dr_w, dr_e, dr_i,
     z_c, z_s, z_n, dz_s, dz_n, dz_j) = build_geometry_faces(r, z)

    sigP_e, sigPar_n = build_face_conductivities(sigma_P, sigma_parallel)

    if mask is not None:
        mu = (mask.astype(np.uint8) != 0)
        face_r = (mu[:-1, :] & mu[1:, :]).astype(float)     # (Nr-1, Nz)
        face_z = (mu[:, :-1] & mu[:,  1:]).astype(float)    # (Nr, Nz-1)
    else:
        face_r = 1.0
        face_z = 1.0

    sigP_e   = sigP_e   * face_r
    sigPar_n = sigPar_n * face_z

    # --- helpers for padding face quantities to cell centers ---
    def pad_east_faces(F_e):       # (Nr-1, Nz) -> (Nr, Nz), east-face flux for each cell
        G = np.zeros((Nr, Nz), dtype=F_e.dtype)
        G[:-1, :] = F_e
        return G

    def pad_west_faces(F_e):       # west-face flux for each cell (east flux of i-1 cell)
        G = np.zeros((Nr, Nz), dtype=F_e.dtype)
        G[1:, :] = F_e
        return G

    def pad_north_faces(F_n):      # (Nr, Nz-1) -> (Nr, Nz), north-face flux for each cell
        G = np.zeros((Nr, Nz), dtype=F_n.dtype)
        G[:, :-1] = F_n
        return G

    def pad_south_faces(F_n):      # south-face flux for each cell (north flux of j-1 cell)
        G = np.zeros((Nr, Nz), dtype=F_n.dtype)
        G[:, 1:] = F_n
        return G

    # --- face-centered 1/(e ne) and pe gradients ---
    # radial faces (between i and i+1)
    ne_e = 0.5 * (ne[:-1, :] + ne[1:, :])
    inv_e_ne_e = 1.0 / (echarge * np.maximum(ne_e, ne_floor))
    dpdr_e = (pe[1:, :] - pe[:-1, :]) / (dr_e[:-1][:, None]) * face_r  # (Nr-1, Nz)

    # axial faces (between j and j+1)
    ne_n = 0.5 * (ne[:, :-1] + ne[:, 1:])
    inv_e_ne_n = 1.0 / (echarge * np.maximum(ne_n, ne_floor))
    dpdz_n = (pe[:, 1:] - pe[:, :-1]) / (dz_n[:-1][None, :]) * face_z  # (Nr, Nz-1)

    # --- "pressure battery" + neutral-rotation (Bz uθ) fluxes ---
    # radial fluxes: F_r = -σP*(1/e ne)*∂r pe - σP*(Bz uθ)
    Fr = -sigP_e * inv_e_ne_e * dpdr_e
    if (Bz is not None) and (un_theta is not None):
        Bz_e = 0.5 * (Bz[:-1, :] + Bz[1:, :])
        u_e  = 0.5 * (un_theta[:-1, :] + un_theta[1:, :])
        Fr -= sigP_e * (Bz_e * u_e)

    # axial fluxes: F_z = -σ||*(1/e ne)*∂z pe
    Fz = -sigPar_n * inv_e_ne_n * dpdz_n

    if mask is not None:
        mu8 = (mask.astype(np.uint8) != 0)
        # radial faces exist only if both adjacent cells are plasma
        face_r = (mu8[:-1,:] & mu8[1:,:]).astype(Fr.dtype)     # (Nr-1,Nz)
        # axial faces exist only if both adjacent cells are plasma
        face_z = (mu8[:, :-1] & mu8[:,  1:]).astype(Fz.dtype)  # (Nr,Nz-1)
        Fr *= face_r
        Fz *= face_z

    # --- divergence of those fluxes (axisymmetric in r) ---
    # radial: (1/r) ∂_r( r Fr ) -> ( r_e*Fr_e - r_w*Fr_w ) / (r_c*dr_i)
    Fr_e_pad = pad_east_faces(Fr)
    Fr_w_pad = pad_west_faces(Fr)
    S_r = (r_e[:, None] * Fr_e_pad - r_w[:, None] * Fr_w_pad) / (np.maximum(r_c, eps)[:, None] * (dr_i[:, None] + eps))

    # axial: ∂_z(Fz) -> ( Fz_north - Fz_south ) / dz_j
    Fz_n_pad = pad_north_faces(Fz)
    Fz_s_pad = pad_south_faces(Fz)
    S_z = (Fz_n_pad - Fz_s_pad) / (dz_j[None, :] + eps)

    # --- ion current divergence ---
    # face-average the supplied cell-centered currents
    if Ji_r is not None:
        Jir_e = 0.5 * (Ji_r[1:, :] + Ji_r[:-1, :]) * face_r # (Nr-1, Nz)
    else:
        Jir_e = np.zeros((Nr-1, Nz), dtype=sigma_P.dtype)

    if Ji_z is not None:
        Jiz_n = 0.5 * (Ji_z[:, 1:] + Ji_z[:, :-1]) * face_z   # (Nr, Nz-1)
    else:
        Jiz_n = np.zeros((Nr, Nz-1), dtype=sigma_P.dtype)

    if mask is not None:
        Jir_e *= face_r
        Jiz_n *= face_z

    Jir_e_pad = pad_east_faces(Jir_e)
    Jir_w_pad = pad_west_faces(Jir_e)  # west = east of left cell
    S_Jr = -(r_e[:, None] * Jir_e_pad - r_w[:, None] * Jir_w_pad) / (np.maximum(r_c, eps)[:, None] * (dr_i[:, None] + eps))

    Jiz_n_pad = pad_north_faces(Jiz_n)
    Jiz_s_pad = pad_south_faces(Jiz_n)
    S_Jz = -(Jiz_n_pad - Jiz_s_pad) / (dz_j[None, :] + eps)

    # total source
    S = S_r + S_z + S_Jr + S_Jz
    if mask is not None:
        S = np.where(mu8, S, 0.0)

    return S

@njit(parallel=True, fastmath=True, cache=True)
def sor_solve(phi, aP, aE, aW, aN, aS, b, mask_u8, omega, max_iter, tol):
    Nr, Nz = phi.shape
    inv_aP = np.zeros_like(aP)
    for j in range(Nz):
        for i in range(Nr):
            inv_aP[i, j] = 0.0 if mask_u8[i, j] == 0 else 1.0 / (aP[i, j] + 1e-30)

    for it in range(max_iter):
        # RED pass
        for j in prange(Nz):
            start_i = 0 if (j & 1) == 0 else 1
            for i in range(start_i, Nr, 2):
                if mask_u8[i, j] == 0:  # skip solids
                    continue
                phiE = phi[i+1, j] if i+1 < Nr else 0.0
                phiW = phi[i-1, j] if i-1 >= 0 else 0.0
                phiN = phi[i, j+1] if j+1 < Nz else 0.0
                phiS = phi[i, j-1] if j-1 >= 0 else 0.0

                rhs     = aE[i, j]*phiE + aW[i, j]*phiW + aN[i, j]*phiN + aS[i, j]*phiS + b[i, j]
                phi_new = rhs * inv_aP[i, j]
                phi[i, j] += omega * (phi_new - phi[i, j])

        # BLACK pass
        for j in prange(Nz):
            start_i = 1 if (j & 1) == 0 else 0
            for i in range(start_i, Nr, 2):
                if mask_u8[i, j] == 0:
                    continue
                phiE = phi[i+1, j] if i+1 < Nr else 0.0
                phiW = phi[i-1, j] if i-1 >= 0 else 0.0
                phiN = phi[i, j+1] if j+1 < Nz else 0.0
                phiS = phi[i, j-1] if j-1 >= 0 else 0.0

                rhs     = aE[i, j]*phiE + aW[i, j]*phiW + aN[i, j]*phiN + aS[i, j]*phiS + b[i, j]
                phi_new = rhs * inv_aP[i, j]
                phi[i, j] += omega * (phi_new - phi[i, j])

        # residual
        max_res = 0.0
        for j in range(Nz):
            for i in range(Nr):
                if mask_u8[i, j] == 0:
                    continue
                phiE = phi[i+1, j] if i+1 < Nr else 0.0
                phiW = phi[i-1, j] if i-1 >= 0 else 0.0
                phiN = phi[i, j+1] if j+1 < Nz else 0.0
                phiS = phi[i, j-1] if j-1 >= 0 else 0.0
                res = abs(aP[i, j]*phi[i, j] - (aE[i, j]*phiE + aW[i, j]*phiW + aN[i, j]*phiN + aS[i, j]*phiS + b[i, j]))
                if res > max_res:
                    max_res = res

        if max_res < tol:
            return it+1, max_res

    return max_iter, max_res


def solve_anisotropic_poisson_FV(geom,
                                 sigma_P, sigma_parallel,
                                 ne=None, pe=None, Bz=None, un_theta=None,
                                 Ji_r=None, Ji_z=None,
                                 ne_floor=1.0,
                                 dphi_dz_cathode_top=None,  # NEW: array (Nr,) at z=zmax_cathode
                                 phi_anode_value=0.0,
                                 phi0=None, omega=1.8, tol=1e-10, max_iter=50_000,
                                 verbose=True):
    """
    Finite-volume SOR for:
      (1/r) ∂_r ( r σ_P ∂_r φ ) + ∂_z ( σ_|| ∂_z φ ) = S   (S from compute_source_S)

    BCs (from geom and masks):
      - Axis r=0: symmetry ∂φ/∂r=0
      - z=0 outside of cathode: ∂φ/∂z=0
      - z=zmax: ∂φ/∂z=0
      - Cathode vertical face: ∂φ/∂r = 0
      - Cathode top (z=zmax_cathode): ∂φ/∂z = dphi_dz_cathode_top[i]
      - Any plasma↔anode face (anode1 or anode2): φ = phi_anode_value (Dirichlet)
      - r=rmax: Neumann (z<zmin_anode), Dirichlet φ=phi_anode_value (z≥zmin_anode)
    """
    Nr, Nz = geom.Nr, geom.Nz

    # RHS source (unchanged physics)
    S = compute_source_S(geom.r, geom.z,
                         sigma_P, sigma_parallel,
                         ne, pe,
                         ne_floor,
                         Bz=Bz, un_theta=un_theta,
                         Ji_r=Ji_r, Ji_z=Ji_z,
                         mask=geom.mask)

    aP, aE, aW, aN, aS, b = assemble_coefficients(
        geom.r, geom.z,
        sigmaP=sigma_P, sigmaPar=sigma_parallel,
        geom=geom,
        dphi_dz_cathode_top=dphi_dz_cathode_top,
        phi_anode_value=phi_anode_value,
        S=S
    )

    phi = np.zeros((Nr, Nz), dtype=sigma_P.dtype) if phi0 is None else np.array(phi0, dtype=sigma_P.dtype, copy=True)

    # SOR with mask skip
    iters, res = sor_solve(phi, aP, aE, aW, aN, aS, b, geom.mask.astype(np.uint8), omega=omega, max_iter=max_iter, tol=tol)

    if verbose:
        print(f"[FV-SOR] iterations = {iters}, residual = {res:.3e}")

    return phi, {"iterations": iters, "residual": float(res)}