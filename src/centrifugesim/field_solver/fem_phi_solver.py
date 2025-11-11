from __future__ import annotations
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, geometry
from dolfinx.fem.petsc import LinearProblem

from centrifugesim import constants

# ---- module-level caches (cleared automatically if mesh changes) ----
def _get_mesh_key(mesh) -> int:
    # Use the cpp mesh pointer as a stable key across Python objects referring
    # to the same mesh. Fallback to id(mesh) if cpp pointer is unavailable.
    try:
        return mesh._cpp_object.__int__()  # unique address (dolfinx.C++)
    except Exception:
        return id(mesh)

# store samplers per (mesh_key, Nr, Nz)
_RECT_SAMPLERS: dict[tuple[int, int, int], "RectSampler"] = {}
# store rect->FE assigners per (V_key, r_id, z_id)
_RECT2FE_CACHE: dict[tuple[int, int, int], "RectToFEInterpolator"] = {}

def clear_fem_sampling_cache():
    """Call this if you rebuild the mesh."""
    _RECT_SAMPLERS.clear()
    _RECT2FE_CACHE.clear()


class RectSampler:
    """Precomputed plan to evaluate CG1 scalar Functions on a fixed (Nr,Nz) grid."""
    def __init__(self, geom, Nr: int | None, Nz: int | None):
        msh = geom.fem.mesh
        tdim = msh.topology.dim

        self.r_axis = geom.r if Nr is None else np.linspace(geom.rmin, geom.rmax, Nr)
        self.z_axis = geom.z if Nz is None else np.linspace(geom.zmin, geom.zmax, Nz)
        RR, ZZ = np.meshgrid(self.r_axis, self.z_axis, indexing="ij")

        # points in 3D coordinates (API expects length-3)
        self.pts3 = np.zeros((RR.size, 3), dtype=np.float64)
        eps = 1e-14
        self.pts3[:, 0] = np.clip(RR.ravel(), geom.rmin + eps, geom.rmax - eps)
        self.pts3[:, 1] = np.clip(ZZ.ravel(), geom.zmin + eps, geom.zmax - eps)

        # locate cells ONCE
        try:
            bbt = geometry.bb_tree(msh, tdim)  # newer dolfinx
        except AttributeError:
            bbt = geometry.BoundingBoxTree(msh, tdim, msh.geometry.x)

        cands = geometry.compute_collisions_points(bbt, self.pts3)
        coll  = geometry.compute_colliding_cells(msh, cands, self.pts3)

        num_pts = self.pts3.shape[0]
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

        self.mask     = cells >= 0
        self.cells_in = cells[self.mask]

        # fixed shapes for fast reuse
        self.num_pts = num_pts
        self.Nr = len(self.r_axis)
        self.Nz = len(self.z_axis)

        # preallocate a values buffer reused across fields (no allocations)
        self.vals = np.zeros((self.cells_in.shape[0], 1), dtype=np.float64)

    def sample(self, func: fem.Function) -> np.ndarray:
        """Evaluate 'func' on the rect grid in ~O(#points)."""
        out = np.full(self.num_pts, np.nan, dtype=np.float64)
        func.eval(self.pts3[self.mask], self.cells_in, self.vals)  # (points, cells, values)
        out[self.mask] = self.vals[:, 0]
        return out.reshape(self.Nr, self.Nz)


class ParallelRectSampler:
    """MPI-parallel rectilinear sampler for CG1 scalar Functions.

    Each rank precomputes which rect-grid points fall into its local cells
    (via BBTree+collisions). Sampling then does a local Function.eval and two
    Allreduces to assemble the global (Nr,Nz) arrays on all ranks (or only root).
    """
    def __init__(self, geom, Nr: int | None, Nz: int | None):
        msh = geom.fem.mesh
        comm = msh.comm
        tdim = msh.topology.dim

        self.comm = comm
        self.rank = comm.rank

        # rect grid axes (same on all ranks)
        self.r_axis = geom.r if Nr is None else np.linspace(geom.rmin, geom.rmax, Nr)
        self.z_axis = geom.z if Nz is None else np.linspace(geom.zmin, geom.zmax, Nz)
        RR, ZZ = np.meshgrid(self.r_axis, self.z_axis, indexing="ij")

        # query points in 3D coords (eval expects length-3)
        self.pts3 = np.zeros((RR.size, 3), dtype=np.float64)
        eps = 1e-14
        self.pts3[:, 0] = np.clip(RR.ravel(), geom.rmin + eps, geom.rmax - eps)
        self.pts3[:, 1] = np.clip(ZZ.ravel(), geom.zmin + eps, geom.zmax - eps)

        # locate local cells only
        try:
            bbt = geometry.bb_tree(msh, tdim)
        except AttributeError:
            bbt = geometry.BoundingBoxTree(msh, tdim, msh.geometry.x)

        cands = geometry.compute_collisions_points(bbt, self.pts3)
        coll  = geometry.compute_colliding_cells(msh, cands, self.pts3)

        num_pts = self.pts3.shape[0]
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

        self.mask_local     = cells >= 0           # points owned by this rank
        self.cells_in_local = cells[self.mask_local]

        # shapes / buffers
        self.num_pts = num_pts
        self.Nr = len(self.r_axis); self.Nz = len(self.z_axis)
        self.vals_local = np.zeros((self.cells_in_local.shape[0], 1), dtype=np.float64)
        # reduction buffers
        self.buf_val  = np.zeros(self.num_pts, dtype=np.float64)
        self.buf_mask = np.zeros(self.num_pts, dtype=np.float64)  # 0/1 flags

    def sample(self, func: fem.Function, *, root: int | None = None) -> np.ndarray | None:
        """Evaluate func in parallel and assemble a global (Nr,Nz) array.
           If root is None, the full array is returned on every rank.
           If root is an int, return the array only on that rank; others return None.
        """
        # local evaluate
        self.buf_val[:]  = 0.0
        self.buf_mask[:] = 0.0
        if self.cells_in_local.size:
            func.eval(self.pts3[self.mask_local], self.cells_in_local, self.vals_local)
            local_vals = self.vals_local[:, 0]
            self.buf_val[self.mask_local]  = local_vals
            self.buf_mask[self.mask_local] = 1.0

        # Allreduce sum (disjoint ownership → sums are safe)
        glob_val  = np.empty_like(self.buf_val)
        glob_mask = np.empty_like(self.buf_mask)
        self.comm.Allreduce(self.buf_val,  glob_val,  op=MPI.SUM)
        self.comm.Allreduce(self.buf_mask, glob_mask, op=MPI.SUM)

        # avoid division-by-zero; put NaN where no rank owned the point
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(glob_mask > 0.0, glob_val / np.maximum(glob_mask, 1.0), np.nan)

        if root is not None and self.rank != root:
            return None
        return out.reshape(self.Nr, self.Nz)


def _get_rect_sampler(geom, Nr: int | None, Nz: int | None, *, parallel: bool = False):
    """Extended factory: choose serial or MPI-parallel sampler and cache."""
    key = (_get_mesh_key(geom.fem.mesh), int(Nr or geom.Nr), int(Nz or geom.Nz), int(bool(parallel)))
    sampler = _RECT_SAMPLERS.get(key)
    if sampler is None:
        sampler = (ParallelRectSampler if parallel else RectSampler)(geom, Nr, Nz)
        _RECT_SAMPLERS[key] = sampler
    return sampler


class RectToFEInterpolator:
    """Precompute bilinear weights mapping rect grid -> CG1 dofs of V."""
    def __init__(self, V: fem.FunctionSpace, r_axis: np.ndarray, z_axis: np.ndarray):
        coords = V.tabulate_dof_coordinates()
        rr = coords[:, 0]; zz = coords[:, 1]

        Nr, Nz = len(r_axis), len(z_axis)
        rmin, rmax = r_axis[0], r_axis[-1]
        zmin, zmax = z_axis[0], z_axis[-1]
        dr = (rmax - rmin) / (Nr - 1)
        dz = (zmax - zmin) / (Nz - 1)

        fr = (rr - rmin) / dr
        fz = (zz - zmin) / dz

        i0 = np.clip(np.floor(fr).astype(np.int32), 0, Nr - 2)
        j0 = np.clip(np.floor(fz).astype(np.int32), 0, Nz - 2)
        tr = fr - i0
        tz = fz - j0

        self.i0, self.j0 = i0, j0
        self.i1, self.j1 = i0 + 1, j0 + 1
        self.tr, self.tz = tr, tz
        self.V_key = id(V)

    def assign(self, func: fem.Function, grid: np.ndarray):
        g00 = grid[self.i0, self.j0]
        g10 = grid[self.i1, self.j0]
        g01 = grid[self.i0, self.j1]
        g11 = grid[self.i1, self.j1]
        tr, tz = self.tr, self.tz
        vals = ((1 - tr)*(1 - tz)*g00 +
                (    tr)*(1 - tz)*g10 +
                (1 - tr)*(    tz)*g01 +
                (    tr)*(    tz)*g11).astype(func.x.array.dtype, copy=False)
        func.x.array[:] = vals
        func.x.scatter_forward()


def _get_rect2fe_assigner(geom_or_axes, V: fem.FunctionSpace) -> RectToFEInterpolator:
    """
    Return a cached rect->FE bilinear assigner keyed by (mesh, r_axis, z_axis).
    geom_or_axes: any object with .r and .z (e.g., Geometry), or the actual arrays.
    """
    # Accept either a geometry-like object or (r_axis, z_axis) via duck typing
    try:
        r_axis = geom_or_axes.r
        z_axis = geom_or_axes.z
    except AttributeError:
        # If a tuple/namespace with .r/.z wasn't passed
        r_axis, z_axis = geom_or_axes

    # Use the mesh as the stable key (so new V instances still hit the cache)
    try:
        m = V.mesh
    except AttributeError:
        # Older dolfinx fallback
        m = V._mesh
    mesh_key = _get_mesh_key(m)
    key = (mesh_key, id(r_axis), id(z_axis))

    obj = _RECT2FE_CACHE.get(key)
    if obj is None:
        obj = RectToFEInterpolator(V, r_axis, z_axis)
        _RECT2FE_CACHE[key] = obj
    return obj

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
    Fast path: uses precomputed bilinear weights for this (mesh, r_axis, z_axis).
    Fallback: vectorized bilinear if cache is missing.
    """
    V = func.function_space
    try:
        # Use the mesh-based cache with the exact axes you passed
        assigner = _get_rect2fe_assigner((r_axis, z_axis), V)
        assigner.assign(func, grid)
        return
    except Exception:
        # Fallback (still vectorized)
        Nr, Nz = grid.shape
        rmin, rmax = r_axis[0], r_axis[-1]
        zmin, zmax = z_axis[0], z_axis[-1]
        dr = (rmax - rmin) / (Nr - 1)
        dz = (zmax - zmin) / (Nz - 1)
        dof_xyz = V.tabulate_dof_coordinates()
        rr = dof_xyz[:, 0]
        zz = dof_xyz[:, 1]
        fr = (rr - rmin) / dr
        fz = (zz - zmin) / dz
        i0 = np.clip(np.floor(fr).astype(np.int32), 0, Nr - 2)
        j0 = np.clip(np.floor(fz).astype(np.int32), 0, Nz - 2)
        i1 = i0 + 1; j1 = j0 + 1
        tr = np.clip(fr - i0, 0.0, 1.0)
        tz = np.clip(fz - j0, 0.0, 1.0)
        g00 = grid[i0, j0]; g10 = grid[i1, j0]; g01 = grid[i0, j1]; g11 = grid[i1, j1]
        vals = ((1-tr)*(1-tz)*g00 + tr*(1-tz)*g10 + (1-tr)*tz*g01 + tr*tz*g11).astype(func.x.array.dtype, copy=False)
        func.x.array[:] = vals
        func.x.scatter_forward()


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


# Initialize FEM coefficient functions (once)
def init_phi_coeffs(geom):
    """
    Create and return a dict of FEM coefficient Functions (pe, ne, inv_e_ne, σ's, u_n, Bz)
    on the CG1 space tied to geom.fem.mesh. Values are zero-initialized; you update them
    with update_phi_coeffs_from_grids(...) each step.
    """
    assert hasattr(geom, "fem"), "Geometry has no FEM mesh. Call geom.build_fem_mesh() first."
    V = fem.functionspace(geom.fem.mesh, ("CG", 1))

    def F(name): return fem.Function(V, name=name)

    coeffs = {
        "grad_pe_r": F("grad_pe_r"),
        "grad_pe_z": F("grad_pe_z"),
        "ne": F("ne"),
        "inv_e_ne": F("inv_e_ne"),
        "sigma_parallel": F("sigma_parallel"),
        "sigma_P": F("sigma_P"),
        "sigma_H": F("sigma_H"),
        "un_r": F("un_r"),
        "un_theta": F("un_theta"),
        "Bz": F("Bz"),
    }
    # Zero-init is fine; you’ll update right away
    for f in coeffs.values():
        f.x.array[:] = 0.0
        f.x.scatter_forward()
    return coeffs


def update_phi_coeffs_from_grids(
    geom, coeffs: dict,
    *,
    ne_grid: np.ndarray | None = None,
    grad_pe_grid_r: np.ndarray | None = None,
    grad_pe_grid_z: np.ndarray | None = None,
    sigma_parallel_grid: np.ndarray | None = None,
    sigma_P_grid: np.ndarray | None = None,
    sigma_H_grid: np.ndarray | None = None,
    un_r_grid: np.ndarray | None = None,
    un_theta_grid: np.ndarray | None = None,
    Bz_grid: np.ndarray | None = None,
):
    """
    Update the given FEM coefficient Functions in-place from rectangular grids.
    Uses the cached bilinear assigner; no point-location is performed here.
    """
    # --- pressure / electron --- #
    _assign_from_rect_grid(coeffs["grad_pe_r"], grad_pe_grid_r, geom.r, geom.z)
    _assign_from_rect_grid(coeffs["grad_pe_z"], grad_pe_grid_z, geom.r, geom.z)

    QE, tiny = constants.q_e, 1e-300
    ne_grid_local = ne_grid

    _assign_from_rect_grid(coeffs["ne"], ne_grid_local, geom.r, geom.z)
    _assign_from_rect_grid(coeffs["inv_e_ne"], 1.0 / (QE * (ne_grid_local + tiny)), geom.r, geom.z)

    _assign_from_rect_grid(coeffs["sigma_parallel"], sigma_parallel_grid, geom.r, geom.z)
    _assign_from_rect_grid(coeffs["sigma_P"],        sigma_P_grid,        geom.r, geom.z)
    _assign_from_rect_grid(coeffs["sigma_H"],        sigma_H_grid,        geom.r, geom.z)

    # --- flow & B (zero if not provided) --- #
    zero = np.zeros_like(ne_grid)
    _assign_from_rect_grid(coeffs["un_r"],     zero if un_r_grid     is None else un_r_grid,     geom.r, geom.z)
    _assign_from_rect_grid(coeffs["un_theta"], zero if un_theta_grid is None else un_theta_grid, geom.r, geom.z)
    _assign_from_rect_grid(coeffs["Bz"],       zero if Bz_grid       is None else Bz_grid,       geom.r, geom.z)


def solve_phi_axisym(geom,
                     coeffs: dict,
                     *,
                     Jz0: float = -2.0e1,
                     sigma_r: float = 5.0e-3,
                     phi_a: float = 0.0,
                     phi_guess: fem.Function | None = None,
                     proj_opts: dict | None = None):
    """
    Solve using FEM coefficient Functions already stored in `coeffs`.
    Optional `phi_guess` (Function on the same V) is used as a PETSc initial guess.
    Returns KSP iteration count as 'ksp_iterations' in the output dict.
    """
    msh        = geom.fem.mesh
    facet_tags = geom.fem.facet_tags
    tags       = geom.fem.tags
    dx         = geom.fem.dx
    ds         = geom.fem.ds

    V = coeffs["ne"].function_space
    x = ufl.SpatialCoordinate(msh)
    r = x[0]

    # Unpack coeffs
    grad_pe_r = coeffs["grad_pe_r"]; grad_pe_z = coeffs["grad_pe_z"]; ne = coeffs["ne"]; inv_e_ne = coeffs["inv_e_ne"]
    sigma_parallel = coeffs["sigma_parallel"]; sigma_P = coeffs["sigma_P"]; sigma_H = coeffs["sigma_H"]
    un_r = coeffs["un_r"]; un_theta = coeffs["un_theta"]; Bz = coeffs["Bz"]

    # Sources/tensor
    S_r = inv_e_ne*grad_pe_r + Bz*un_theta
    S_z = inv_e_ne*grad_pe_z
    J_S = ufl.as_vector([sigma_P*S_r + sigma_H*Bz*un_r, sigma_parallel*S_z])
    K   = ufl.as_matrix([[sigma_P, 0.0], [0.0, sigma_parallel]])

    # Weak forms
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(K*ufl.grad(u), ufl.grad(v)) * r * dx
    L = ufl.inner(J_S, ufl.grad(v))            * r * dx

    # Neumann on cathode top
    Jz_profile = Jz0 * ufl.exp(-0.5*(r/sigma_r)**2)
    L += (Jz_profile * v) * r * ds(tags["CATH_TOP"])

    # Dirichlet on anode edges + right wall (mid, top)
    bcs = []
    phi_a_fun = fem.Function(V); phi_a_fun.x.array[:] = phi_a
    for key in ("ANODE_ISL", "RIGHT_MID", "RIGHT_TOP"):
        facets = facet_tags.find(tags[key])
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
            if dofs.size > 0:
                bcs.append(fem.dirichletbc(phi_a_fun, dofs))

    # Prepare solution holder and PETSc options
    phi_u = fem.Function(V, name="phi")
    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-5}
    if phi_guess is not None:
        # Copy user's guess to the solution vector and enable nonzero initial guess
        if phi_guess.function_space is V:
            phi_u.x.array[:] = phi_guess.x.array
            phi_u.x.scatter_forward()
        else:
            # fallback: copy as array if sizes match (same mesh/space assumed by user)
            phi_u.x.array[:] = phi_guess.x.array
            phi_u.x.scatter_forward()
        petsc_opts["ksp_initial_guess_nonzero"] = True
    
    # Solve with LinearProblem (and pass 'u=phi_u' so KSP uses our vector)
    problem = LinearProblem(a, L, bcs=bcs, u=phi_u, petsc_options=petsc_opts)
    phi = problem.solve()

    # Try to fetch KSP iterations
    ksp_its = -1
    for attr in ("solver", "ksp", "_solver"):
        if hasattr(problem, attr):
            try:
                ksp_its = getattr(problem, attr).getIterationNumber()
                break
            except Exception:
                pass

    # Projections
    def _project(expr, name: str):
        up, vp = ufl.TrialFunction(V), ufl.TestFunction(V)
        aP = ufl.inner(up, vp) * dx
        LP = ufl.inner(expr, vp) * dx
        opts = {"ksp_type": "cg", "pc_type": "jacobi"} if proj_opts is None else proj_opts
        proj = LinearProblem(aP, LP, petsc_options=opts)
        out = fem.Function(V, name=name)
        out.x.array[:] = proj.solve().x.array
        return out

    grad_phi = ufl.grad(phi)
    Er = _project(-grad_phi[0], "Er")
    Ez = _project(-grad_phi[1], "Ez")
    Jr = _project(-(sigma_P * grad_phi[0]) + (sigma_P * S_r + sigma_H * Bz * un_r), "Jr")
    Jz = _project(-(sigma_parallel * grad_phi[1]) + (sigma_parallel * S_z),          "Jz")

    # Ohmic heating in neutral frame
    Eprime_r = -grad_phi[0] + S_r
    Eprime_z = -grad_phi[1] + S_z
    q_ohm = _project(sigma_P*Eprime_r*Eprime_r + sigma_parallel*Eprime_z*Eprime_z, "q_ohm")

    # Diagnostics
    two_pi = 2*np.pi
    n = ufl.FacetNormal(msh)
    Jtot = K*ufl.grad(phi) - J_S
    I_applied  = fem.assemble_scalar(fem.form(two_pi * r * Jz_profile         * ds(tags["CATH_TOP"])))
    I_from_sol = fem.assemble_scalar(fem.form(two_pi * r * ufl.inner(Jtot, n) * ds(tags["CATH_TOP"])))

    return {
        "V": V,
        "ne": ne,
        "phi": phi,
        "Er": Er, "Ez": Ez,
        "Jr": Jr, "Jz": Jz,
        "q_ohm": q_ohm,
        "coeffs": coeffs,
        "integrals": {"I_applied": I_applied, "I_from_solution": I_from_sol},
        "ksp_iterations": ksp_its,
    }


def solve_phi_axisym_from_grids(
    geom, *,
    # required driving fields (Nr×Nz); provide either pe_grid or (ne_grid and Te_grid)
    pe_grid: np.ndarray | None = None,
    ne_grid: np.ndarray | None = None,
    Te_grid: np.ndarray | None = None,
    # optional transport fields
    sigma_parallel_grid: np.ndarray | None = None,
    sigma_P_grid: np.ndarray | None = None,
    sigma_H_grid: np.ndarray | None = None,
    # flow and B
    un_r_grid: np.ndarray | None = None,
    un_theta_grid: np.ndarray | None = None,
    Bz_grid: np.ndarray | None = None,
    # BC parameters
    Jz0: float = -2.0e1,
    sigma_r: float = 5.0e-3,
    phi_a: float = 0.0,
    # optional initial guess (Function on the same mesh/space)
    phi_guess: fem.Function | None = None,
    # projection solver options
    proj_opts: dict | None = None,
):
    """
    Same as before, but accepts optional `phi_guess` and returns 'ksp_iterations'.
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
    def F(name): return fem.Function(V, name=name)

    # Pressure/electron properties
    if pe_grid is None:
        if (ne_grid is None) or (Te_grid is None):
            raise ValueError("Provide either pe_grid, or both ne_grid and Te_grid.")
        pe_grid = constants.kb * ne_grid * Te_grid

    pe = F("pe"); _assign_from_rect_grid(pe, pe_grid, geom.r, geom.z)
    ne = F("ne"); _assign_from_rect_grid(ne, ne_grid, geom.r, geom.z)

    sigma_parallel = F("sigma_parallel"); _assign_from_rect_grid(sigma_parallel, sigma_parallel_grid, geom.r, geom.z)
    sigma_P        = F("sigma_P");        _assign_from_rect_grid(sigma_P,        sigma_P_grid,        geom.r, geom.z)
    sigma_H        = F("sigma_H");        _assign_from_rect_grid(sigma_H,        sigma_H_grid,        geom.r, geom.z)

    un_r     = F("un_r");     _assign_from_rect_grid(un_r,     np.zeros_like(pe_grid) if un_r_grid     is None else un_r_grid,     geom.r, geom.z)
    un_theta = F("un_theta"); _assign_from_rect_grid(un_theta, np.zeros_like(pe_grid) if un_theta_grid is None else un_theta_grid, geom.r, geom.z)
    Bz       = F("Bz");       _assign_from_rect_grid(Bz,       np.zeros_like(pe_grid) if Bz_grid       is None else Bz_grid,       geom.r, geom.z)

    QE = constants.q_e; tiny = 1e-300
    inv_e_ne = F("inv_e_ne"); _assign_from_rect_grid(inv_e_ne, 1.0/(QE*(ne_grid + tiny)), geom.r, geom.z)

    # Sources/tensor
    grad_pe = ufl.grad(pe)
    S_r = - inv_e_ne*grad_pe[0] + Bz*un_theta
    S_z = - inv_e_ne*grad_pe[1]
    J_S = ufl.as_vector([sigma_P*S_r + sigma_H*Bz*un_r, sigma_parallel*S_z])
    K   = ufl.as_matrix([[sigma_P, 0.0], [0.0, sigma_parallel]])

    # Weak forms
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(K*ufl.grad(u), ufl.grad(v)) * r * dx
    L = ufl.inner(J_S, ufl.grad(v))            * r * dx

    # BCs
    Jz_profile = Jz0 * ufl.exp(-0.5*(r/sigma_r)**2)
    L += ( Jz_profile * v ) * r * ds(tags["CATH_TOP"])

    bcs = []
    phi_a_fun = fem.Function(V); phi_a_fun.x.array[:] = phi_a
    for key in ("ANODE_ISL", "RIGHT_MID", "RIGHT_TOP"):
        facets = facet_tags.find(tags[key])
        if facets.size > 0:
            dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
            if dofs.size > 0:
                bcs.append(fem.dirichletbc(phi_a_fun, dofs))

    # Initial guess + PETSc options
    phi_u = fem.Function(V, name="phi")
    petsc_opts = {"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-9}
    if phi_guess is not None:
        phi_u.x.array[:] = phi_guess.x.array
        phi_u.x.scatter_forward()
        petsc_opts["ksp_initial_guess_nonzero"] = True

    # Solve
    problem = LinearProblem(a, L, bcs=bcs, u=phi_u, petsc_options=petsc_opts)
    phi = problem.solve()

    # Try to fetch KSP iterations
    ksp_its = -1
    for attr in ("solver", "ksp", "_solver"):
        if hasattr(problem, attr):
            try:
                ksp_its = getattr(problem, attr).getIterationNumber()
                break
            except Exception:
                pass

    # Projections
    def _project(expr, name: str):
        up, vp = ufl.TrialFunction(V), ufl.TestFunction(V)
        aP = ufl.inner(up, vp) * dx
        LP = ufl.inner(expr, vp) * dx
        opts = {"ksp_type": "cg", "pc_type": "jacobi"} if proj_opts is None else proj_opts
        proj = LinearProblem(aP, LP, petsc_options=opts)
        out = fem.Function(V, name=name)
        out.x.array[:] = proj.solve().x.array
        return out

    grad_phi = ufl.grad(phi)
    Er = _project(-grad_phi[0], "Er")
    Ez = _project(-grad_phi[1], "Ez")
    Jr = _project(-(sigma_P * grad_phi[0]) + (sigma_P * S_r + sigma_H * Bz * un_r), "Jr")
    Jz = _project(-(sigma_parallel * grad_phi[1]) + (sigma_parallel * S_z),          "Jz")

    # q_ohm
    Eprime_r = -grad_phi[0] + S_r
    Eprime_z = -grad_phi[1] + S_z
    q_ohm = _project(sigma_P*Eprime_r*Eprime_r + sigma_parallel*Eprime_z*Eprime_z, "q_ohm")

    # Diagnostics
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
        "integrals": {"I_applied": I_applied, "I_from_solution": I_from_sol},
        "ksp_iterations": ksp_its,
    }


def functions_to_rect_grids(geom,
                            fields: dict[str, fem.Function],
                            Nr: int | None = None,
                            Nz: int | None = None,
                            sampler=None,
                            *,
                            parallel: bool | None = None,
                            root: int | None = None) -> dict[str, np.ndarray | None]:
    """
    Sample DOLFINx scalar Functions on the (NrxNz) rect grid.
    - If 'sampler' is a ParallelRectSampler (or parallel=True), evaluate in MPI and assemble via Allreduce.
    - If 'sampler' is a RectSampler (or parallel=False), evaluate on this rank only.
    - If 'root' is not None in parallel mode, only that rank returns the arrays; others return None.
    """
    if sampler is None:
        sampler = _get_rect_sampler(geom, Nr, Nz, parallel=bool(parallel))

    out = {}
    if isinstance(sampler, ParallelRectSampler):
        for name, func in fields.items():
            out[name] = sampler.sample(func, root=root)
    else:
        # serial sampler (your old fast path)
        for name, func in fields.items():
            out[name] = sampler.sample(func)
    return out