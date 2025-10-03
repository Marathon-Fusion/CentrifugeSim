import numpy as np

from centrifugesim import constants

def init_particles_positions_and_weights(nr: int,
                                        r_min: float,
                                        r_max: float,
                                        z_min: float,
                                        z_max: float,
                                        dr: float,
                                        dz: float,
                                        density: float,
                                        w: int = None,
                                        np_per_cell: int = None,
                                        seed: int = None):
    """
    Initialize particle positions and weights for a cylindrical PIC simulation.

    Parameters
    ----------
    np_per_cell : int
        Number of particles per cell.
    weight : int
        weight of macroparticles
    nr : int
        Number of cells in r direction for domain.
    r_min, r_max : float
        Radial bounds for particle initialization.
    z_min, z_max : float
        Axial (z) bounds for particle initialization.
    dr, dz : float
        Grid spacing in the radial and axial directions.
    density : float
        Physical particle density (particles per unit volume).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    Np : int
        Total number of macroparticles initialized.
    r_positions : np.ndarray
        Array of radial positions (shape: Np).
    z_positions : np.ndarray
        Array of axial positions (shape: Np).
    weights : np.ndarray
        Array of particle weights (shape: Np).
    """
    rng = np.random.default_rng(seed)

    r_positions = np.array([])
    z_positions = np.array([])
    weights = np.array([])

    assert not (w is None and np_per_cell is None), "ERROR: Both weight and np_per_cell are None. User needs to define one of either"

    for j in range(nr):
        rj = r_min + (j+0.5) * dr

        if r_min <= rj <= r_max:

            #a = 1.0/3 if j==0 else 1
            a = 1
            volume_j = a*(z_max-z_min)*2*np.pi*rj*dr

            Nj = density*volume_j

            if(np_per_cell is not None):
                Nj_w = int(np_per_cell*(z_max-z_min)/dz)
                wp = np.ones(Nj_w)*int(Nj/Nj_w)

            elif(w is not None):
                Nj_w = int(Nj/w)
                wp = np.ones(Nj_w)*w

            rp = r_min + j*dr + dr * np.sqrt(rng.random(Nj_w))
            zp = z_min + (z_max-z_min) * rng.random(Nj_w)

            weights = np.concatenate((weights,wp))
            r_positions = np.concatenate((r_positions,rp))
            z_positions = np.concatenate((z_positions,zp))

    weights = weights.astype(np.float32)
    Np = weights.shape[0]

    return Np, r_positions, z_positions, weights


def initialize_ions_from_ni_w(ni, Ti, rmesh, zmesh, dr, dz, w, mi, rng=None):
    """
    Build macro-particle positions (r,z) and velocities (vr,vt,vz)
    from nodal ion density/temperature fields in an r–z cylindrical mesh.

    Parameters
    ----------
    ni, Ti : 2-D arrays (Nr, Nz)      Ion density [m⁻³] and temperature [K]
    rmesh, zmesh : 2-D arrays (Nr, Nz)   Nodal coordinates [m]
    dr, dz : float                     Uniform cell sizes [m]
    w : float                          Particle weight (real particles per macro-particle)
    mi : float                         Ion mass [kg]
    rng : np.random.Generator or None  Optional RNG for reproducibility

    Returns
    -------
    r_p, z_p            : 1-D arrays of particle positions
    vr_p, vt_p, vz_p    : 1-D arrays of velocities
    w_p                 : 1-D array (all elements == w)
    """
    if rng is None:
        rng = np.random.default_rng()

    Nr, Nz = ni.shape
    assert Ti.shape == (Nr, Nz) and rmesh.shape == (Nr, Nz) and zmesh.shape == (Nr, Nz)

    # --- number of macro-particles per cell ---------------------------------
    # Cylindrical cell volume centred on nodal (r_i, z_j) :
    #   V = 2π r_i  dr  dz    (use πr²dr dz at r=0)
    vol = 2.0 * np.pi * rmesh * dr * dz
    vol = np.where(rmesh == 0.0, 1/3.*np.pi * dr**2 * dz, vol)

    Np_cell = np.rint(ni * vol / w).astype(int)      # integer macroparticle count
    total_p = Np_cell.sum()
    if total_p == 0:
        raise ValueError("Density/weight combination produced zero macroparticles")

    # --- allocate output ----------------------------------------------------
    r_p  = np.empty(total_p)
    z_p  = np.empty_like(r_p)
    vr_p = np.empty_like(r_p)
    vt_p = np.empty_like(r_p)
    vz_p = np.empty_like(r_p)
    w_p  = np.full(total_p, w)

    # --- generate particles cell-by-cell ------------------------------------
    offset = 0
    for i in range(Nr):
        for j in range(Nz):
            n_here = Np_cell[i, j]
            if n_here == 0:
                continue

            # radial bounds of the cell
            r_ctr = rmesh[i, j]
            r_min = max(r_ctr - 0.5*dr, 0.0)
            r_max = r_ctr + 0.5*dr

            # sample positions uniformly in cell volume
            u_r   = rng.random(n_here)
            r_p[offset:offset+n_here] = np.sqrt(r_min**2 + (r_max**2 - r_min**2) * u_r)
            z_p[offset:offset+n_here] = zmesh[i, j] - 0.5*dz + dz * rng.random(n_here)

            # Maxwellian velocities, independent components
            v_th = np.sqrt(constants.kb * Ti[i, j] / mi)
            vr_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)
            vt_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)
            vz_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)

            offset += n_here

    return w_p, r_p, z_p, vr_p, vt_p, vz_p


def initialize_ions_from_ni_nppc(ni, Ti, rmesh, zmesh, dr, dz, nppc, mi, rng=None):
    """
    Build macro-particle positions and velocities from nodal ion density/temperature
    fields on an r-z cylindrical mesh.

    Parameters
    ----------
    ni, Ti : 2-D arrays (Nr, Nz)
        Ion density [m⁻³] and temperature [K] at mesh nodes.
    rmesh, zmesh : 2-D arrays (Nr, Nz)
        Nodal coordinates [m].
    dr, dz : float
        Uniform cell sizes [m] in r and z.
    nppc : int
        Desired (integer) number of macroparticles per cell that contains plasma.
    mi : float
        Ion mass [kg].
    rng : np.random.Generator or None, optional
        Random-number generator for reproducibility.

    Returns
    -------
    w_p              : 1-D array of particle weights [number of real ions per macroparticle]
    r_p, z_p         : 1-D arrays of particle positions [m]
    vr_p, vt_p, vz_p : 1-D arrays of particle velocities [m s⁻¹]
    """
    if rng is None:
        rng = np.random.default_rng()
    if not isinstance(nppc, int) or nppc <= 0:
        raise ValueError("`nppc` must be a positive integer")

    Nr, Nz = ni.shape
    assert Ti.shape == (Nr, Nz) and rmesh.shape == (Nr, Nz) and zmesh.shape == (Nr, Nz)

    # --- cell volumes -------------------------------------------------------
    vol = 2.0 * np.pi * rmesh * dr * dz            # cylindrical shell volume
    vol = np.where(rmesh == 0.0, (1/3.)*np.pi*dr**2*dz, vol)   # cell touching axis

    # --- macroparticles per cell & per-particle weight ----------------------
    Np_cell = np.where(ni > 0.0, nppc, 0)          # fixed count where density non-zero
    w_cell  = np.zeros_like(ni, dtype=float)
    mask    = Np_cell > 0
    w_cell[mask] = (ni[mask] * vol[mask]) / nppc   # real ions represented by each macro-ion

    total_p = Np_cell.sum()
    if total_p == 0:
        raise ValueError("Chosen `nppc` and density field produced zero macroparticles")

    # --- allocate arrays ----------------------------------------------------
    r_p  = np.empty(total_p)
    z_p  = np.empty_like(r_p)
    vr_p = np.empty_like(r_p)
    vt_p = np.empty_like(r_p)
    vz_p = np.empty_like(r_p)
    w_p  = np.empty_like(r_p)

    # --- generate particles -------------------------------------------------
    offset = 0
    for i in range(Nr):
        for j in range(Nz):
            n_here = Np_cell[i, j]
            if n_here == 0:
                continue

            # radial bounds of the cell
            r_ctr = rmesh[i, j]
            r_min = max(r_ctr - 0.5 * dr, 0.0)
            r_max = r_ctr + 0.5 * dr

            # positions: uniform in cylindrical cell volume
            u_r   = rng.random(n_here)
            r_p[offset:offset+n_here] = np.sqrt(r_min**2 + (r_max**2 - r_min**2) * u_r)
            z_p[offset:offset+n_here] = zmesh[i, j] - 0.5 * dz + dz * rng.random(n_here)

            # Maxwellian velocities
            v_th = np.sqrt(constants.kb * Ti[i, j] / mi)
            vr_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)
            vt_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)
            vz_p[offset:offset+n_here] = rng.normal(0.0, v_th, n_here)

            # per-particle weight for this cell
            w_p[offset:offset+n_here] = w_cell[i, j]

            offset += n_here

    return w_p, r_p, z_p, vr_p, vt_p, vz_p
