from __future__ import annotations
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem

from centrifugesim import constants

# ------------------------------------------------------------
# Helpers: interpolate rectangular Nr×Nz arrays -> FE dofs
# ------------------------------------------------------------
def _bilinear_sample_to_points(grid: np.ndarray,
                               r_axis: np.ndarray,
                               z_axis: np.ndarray,
                               pts_r: np.ndarray,
                               pts_z: np.ndarray) -> np.ndarray:
    """
    Bilinear sample 'grid[i,j]' defined on tensor grid (r_axis, z_axis)
    at arbitrary points (pts_r, pts_z). Returns 1D array (npts,).
    Assumes r_axis and z_axis are sorted increasing and uniform (as in Geometry).
    """
    Nr, Nz = grid.shape
    rmin, rmax = r_axis[0], r_axis[-1]
    zmin, zmax = z_axis[0], z_axis[-1]
    dr = (rmax - rmin) / (Nr - 1)
    dz = (zmax - zmin) / (Nz - 1)

    # fractional indices
    fr = (pts_r - rmin) / dr
    fz = (pts_z - zmin) / dz
    # clamp to valid bilinear cell (Nr-2, Nz-2)
    i0 = np.clip(np.floor(fr).astype(np.int32), 0, Nr - 2)
    j0 = np.clip(np.floor(fz).astype(np.int32), 0, Nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # weights
    tr = np.clip(fr - i0, 0.0, 1.0)
    tz = np.clip(fz - j0, 0.0, 1.0)

    # gather corners
    g00 = grid[i0, j0]
    g10 = grid[i1, j0]
    g01 = grid[i0, j1]
    g11 = grid[i1, j1]

    # bilinear blend
    return ( (1 - tr) * (1 - tz) * g00
           + (    tr) * (1 - tz) * g10
           + (1 - tr) * (    tz) * g01
           + (    tr) * (    tz) * g11 )


def _assign_from_rect_grid(func: fem.Function,
                           grid: np.ndarray,
                           r_axis: np.ndarray,
                           z_axis: np.ndarray):
    """
    Fill a scalar CG1 fem.Function 'func' by bilinearly sampling a rectangular
    NrxNz array 'grid' at the FE dof coordinates (r,z).
    """
    dof_xyz = func.function_space.tabulate_dof_coordinates()
    rr = dof_xyz[:, 0]
    zz = dof_xyz[:, 1]
    vals = _bilinear_sample_to_points(grid, r_axis, z_axis, rr, zz).astype(func.x.array.dtype, copy=False)
    func.x.array[:] = vals
    func.x.scatter_forward()


# ------------------------------------------------------------
# Helpers: FE function -> rectangular grid (robust)
# ------------------------------------------------------------
def _build_point_sampling(geom, Nr: int | None, Nz: int | None):
    """
    Create (pts3, mask, cells_in, shape) to evaluate FE Functions on a
    rectilinear grid. Uses geom.r, geom.z by default.
    """
    msh = geom.fem.mesh
    tdim = msh.topology.dim

    # grid (use geom's grid unless overridden)
    r_axis = geom.r if Nr is None else np.linspace(geom.rmin, geom.rmax, Nr)
    z_axis = geom.z if Nz is None else np.linspace(geom.zmin, geom.zmax, Nz)

    RR, ZZ = np.meshgrid(r_axis, z_axis, indexing="ij")
    pts3 = np.zeros((RR.size, 3), dtype=np.float64)
    # Keep a tiny distance away from exact boundaries to avoid ambiguity
    eps = 1e-14
    pts3[:, 0] = np.clip(RR.ravel(), geom.rmin + eps, geom.rmax - eps)
    pts3[:, 1] = np.clip(ZZ.ravel(), geom.zmin + eps, geom.zmax - eps)

    # locate containing cells
    try:
        bbt = geometry.bb_tree(msh, tdim)          # newer dolfinx
    except AttributeError:
        bbt = geometry.BoundingBoxTree(msh, tdim, msh.geometry.x)  # older

    cands = geometry.compute_collisions_points(bbt, pts3)
    coll  = geometry.compute_colliding_cells(msh, cands, pts3)

    num_pts = pts3.shape[0]
    try:
        data, offsets = coll.array, coll.offsets
        cells = np.full(num_pts, -1, dtype=np.int32)
        starts, ends = offsets[:-1], offsets[1:]
        idx = np.nonzero(ends > starts)[0]
        cells[idx] = data[starts[idx]]
    except AttributeError:
        cells = np.asarray(coll, dtype=np.int32)
        if cells.shape != (num_pts,):
            raise RuntimeError(f"Unexpected colliding-cells shape {cells.shape}")

    mask = cells >= 0
    cells_in = cells[mask]
    return pts3, mask, cells_in, (r_axis, z_axis)


def _sample_scalar_to_grid(geom, func: fem.Function,
                           pts3: np.ndarray,
                           mask: np.ndarray,
                           cells_in: np.ndarray,
                           r_axis: np.ndarray,
                           z_axis: np.ndarray) -> np.ndarray:
    """
    Evaluate scalar fem.Function on the rect grid described by (pts3,mask,cells_in).
    Returns array shape (Nr,Nz) with NaN outside the plasma.
    """
    Nr, Nz = len(r_axis), len(z_axis)
    num_pts = pts3.shape[0]
    out = np.full(num_pts, np.nan, dtype=np.float64)
    vals = np.zeros((cells_in.shape[0], 1), dtype=np.float64)
    # In this dolfinx API: order is (points, cells, values)
    func.eval(pts3[mask], cells_in, vals)
    out[mask] = vals[:, 0]
    return out.reshape(Nr, Nz)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def solve_phi_axisym_from_grids(
    geom, *,
    # required driving fields (Nr×Nz); provide either pe_grid or (ne_grid and Te_grid)
    pe_grid: np.ndarray | None = None,
    ne_grid: np.ndarray | None = None,
    Te_grid: np.ndarray | None = None,
    # optional transport fields; if None, simple defaults from ne are used
    sigma_parallel_grid: np.ndarray | None = None,
    sigma_P_grid: np.ndarray | None = None,
    sigma_H_grid: np.ndarray | None = None,
    # flow and B (defaults to zero flow and uniform Bz=0):
    un_r_grid: np.ndarray | None = None,
    un_theta_grid: np.ndarray | None = None,
    Bz_grid: np.ndarray | None = None,
    # BC parameters
    Jz0: float = -2.0e1,
    sigma_r: float = 5.0e-3,
    phi_a: float = 0.0,
    # projection solver options
    proj_opts: dict | None = None,
):
    """
    Solve for φ on geom.fem.mesh, using axisymmetric weighted form and
    your boundary conditions. All *_grid arrays are defined on geom.r × geom.z.

    Returns a dict with DOLFINx Functions and integral diagnostics.
    """
    assert hasattr(geom, "fem"), "Geometry has no FEM mesh. Call geom.build_fem_mesh() first."

    msh        = geom.fem.mesh
    facet_tags = geom.fem.facet_tags
    tags       = geom.fem.tags
    dx         = geom.fem.dx
    ds         = geom.fem.ds

    V = fem.functionspace(msh, ("CG", 1))
    x = ufl.SpatialCoordinate(msh)
    r = x[0]

    # -- Build coefficient Functions and assign from grids --
    def F(name):
        return fem.Function(V, name=name)

    # Pressure/electron properties
    if pe_grid is None:
        if (ne_grid is None) or (Te_grid is None):
            raise ValueError("Provide either pe_grid, or both ne_grid and Te_grid.")
        pe_grid = constants.kb * ne_grid * Te_grid  # KB * ne * Te

    pe = F("pe"); _assign_from_rect_grid(pe, pe_grid, geom.r, geom.z)
    ne = F("ne"); _assign_from_rect_grid(ne, ne_grid, geom.r, geom.z)

    sigma_parallel = F("sigma_parallel"); _assign_from_rect_grid(sigma_parallel, sigma_parallel_grid, geom.r, geom.z)
    sigma_P        = F("sigma_P");        _assign_from_rect_grid(sigma_P,        sigma_P_grid,        geom.r, geom.z)
    sigma_H        = F("sigma_H");        _assign_from_rect_grid(sigma_H,        sigma_H_grid,        geom.r, geom.z)

    # Flow and B
    un_r     = F("un_r");     _assign_from_rect_grid(un_r,     np.zeros_like(pe_grid) if un_r_grid     is None else un_r_grid,     geom.r, geom.z)
    un_theta = F("un_theta"); _assign_from_rect_grid(un_theta, np.zeros_like(pe_grid) if un_theta_grid is None else un_theta_grid, geom.r, geom.z)
    Bz       = F("Bz");       _assign_from_rect_grid(Bz,       np.zeros_like(pe_grid) if Bz_grid       is None else Bz_grid,       geom.r, geom.z)

    # 1/(e ne): if ne not supplied, infer from pe with a dummy Te to avoid singularities
    QE = constants.q_e
    tiny = 1e-300
    if ne is not None:
        inv_e_ne = F("inv_e_ne"); _assign_from_rect_grid(inv_e_ne, 1.0/(QE*(ne_grid + tiny)), geom.r, geom.z)
    else:
        # fallback: use |pe| to scale, avoid division by zero
        inv_e_ne = F("inv_e_ne"); _assign_from_rect_grid(inv_e_ne, np.full_like(pe_grid, 1.0/(QE*1.0)), geom.r, geom.z)

    # Source terms (neutral frame E'):  S_r = - (∂pe/∂r)/(e ne) + Bz*un_theta ; S_z = - (∂pe/∂z)/(e ne)
    grad_pe = ufl.grad(pe)
    S_r = - inv_e_ne*grad_pe[0] + Bz*un_theta
    S_z = - inv_e_ne*grad_pe[1]
    J_S = ufl.as_vector([sigma_P*S_r + sigma_H*Bz*un_r, sigma_parallel*S_z])
    K   = ufl.as_matrix([[sigma_P, 0.0],
                         [0.0,     sigma_parallel]])

    # ---------- Weak form (axisymmetric) ----------
    u = ufl.TrialFunction(V); v = ufl.TestFunction(V)
    a = ufl.inner(K*ufl.grad(u), ufl.grad(v)) * r * dx
    L = ufl.inner(J_S, ufl.grad(v))            * r * dx

    # Neumann: J_tot·n = Jz_profile on the cathode top
    Jz_profile = Jz0 * ufl.exp(-0.5*(r/sigma_r)**2)
    L += ( Jz_profile * v ) * r * ds(tags["CATH_TOP"])

    # Dirichlet on: anode island edges, right wall between anodes, and right wall above anodes
    bcs = []
    phi_a_fun = fem.Function(V); phi_a_fun.x.array[:] = phi_a

    for key in ("ANODE_ISL", "RIGHT_MID", "RIGHT_TOP"):
        facets = facet_tags.find(tags[key])
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
            if dofs.size > 0:
                bcs.append(fem.dirichletbc(phi_a_fun, dofs))

    # Solve
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-9})
    phi = problem.solve()

    # ---------- Derived fields ----------
    def _project(expr, name: str):
        up, vp = ufl.TrialFunction(V), ufl.TestFunction(V)
        aP = ufl.inner(up, vp) * dx
        LP = ufl.inner(expr, vp) * dx
        opts = {"ksp_type": "cg", "pc_type": "jacobi"}
        proj = LinearProblem(aP, LP, petsc_options=opts if proj_opts is None else proj_opts)
        out = fem.Function(V, name=name)
        out.x.array[:] = proj.solve().x.array
        return out

    grad_phi = ufl.grad(phi)
    Er = _project(-grad_phi[0], "Er")
    Ez = _project(-grad_phi[1], "Ez")

    Jr = _project(-(sigma_P * grad_phi[0]) + (sigma_P * S_r + sigma_H * Bz * un_r), "Jr")
    Jz = _project(-(sigma_parallel * grad_phi[1]) + (sigma_parallel * S_z),          "Jz")

    # E' for heating: E' = E + u_n×B - ∇pe/(e ne)  => components already in S_r, S_z
    Eprime_r = -grad_phi[0] + S_r
    Eprime_z = -grad_phi[1] + S_z
    q_ohm = _project(sigma_P*Eprime_r*Eprime_r + sigma_parallel*Eprime_z*Eprime_z, "q_ohm")

    # ---------- Diagnostics on cathode top ----------
    two_pi = 2.0*np.pi
    n = ufl.FacetNormal(msh)
    Jtot = K*ufl.grad(phi) - J_S

    I_applied = fem.assemble_scalar(fem.form(two_pi * r * Jz_profile         * ds(tags["CATH_TOP"])))
    I_from_sol= fem.assemble_scalar(fem.form(two_pi * r * ufl.inner(Jtot, n) * ds(tags["CATH_TOP"])))

    return {
        "V": V,
        "phi": phi,
        "Er": Er, "Ez": Ez,
        "Jr": Jr, "Jz": Jz,
        "q_ohm": q_ohm,
        "coeffs": {
            "pe": pe, "sigma_parallel": sigma_parallel, "sigma_P": sigma_P, "sigma_H": sigma_H,
            "Bz": Bz, "un_r": un_r, "un_theta": un_theta, "inv_e_ne": inv_e_ne, "ne": ne
        },
        "integrals": {
            "I_applied": I_applied,
            "I_from_solution": I_from_sol
        }
    }


def functions_to_rect_grids(geom,
                            fields: dict[str, fem.Function],
                            Nr: int | None = None,
                            Nz: int | None = None) -> dict[str, np.ndarray]:
    """
    Sample the given DOLFINx scalar Functions on the rectangular (NrxNz) grid
    (defaults to geom.Nr x geom.Nz). Returns {name: array}.
    """
    pts3, mask, cells_in, (r_axis, z_axis) = _build_point_sampling(geom, Nr, Nz)
    out = {}
    for name, func in fields.items():
        arr = _sample_scalar_to_grid(geom, func, pts3, mask, cells_in, r_axis, z_axis)
        out[name] = arr
    return out