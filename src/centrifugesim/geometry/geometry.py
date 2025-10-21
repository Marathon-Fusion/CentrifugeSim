import numpy as np
from typing import Dict

# --- DOLFINx/GMsh physical tags we will use on the FEM mesh ---
G_DOMAIN    = 11
G_CATH_TOP  = 3   # cathode top (Neumann Jz)
G_ANODE_ISL = 5   # internal anode island edges (Dirichlet)
G_RIGHT_MID = 7   # right wall between anodes (Dirichlet)
G_RIGHT_TOP = 8   # right wall above top anode (Dirichlet)

class Geometry:
    """
    2D RZ grid geometry with anode/cathode regions and coil metadata.
    Coils MUST be outside the domain; any overlap with the domain raises.
    """

    def __init__(
        self,
        rmax: float,
        zmin: float,
        zmax: float,
        Nr: int,
        Nz: int,
        rmax_cathode: float,
        zmax_cathode: float,
        rmin_anode: float,
        zmin_anode: float,
        zmax_anode: float,
        zmin_anode2: float,
        temperature_cathode: float,
        temperature_anode: float
    ):
        # Domain limits
        self.rmin = 0.0
        self.rmax = float(rmax)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.Nr = int(Nr)
        self.Nz = int(Nz)

        # Grid spacings
        self.dr = (self.rmax - self.rmin) / (self.Nr - 1)
        self.dz = (self.zmax - self.zmin) / (self.Nz - 1)

        # Coordinates and mesh
        self.r = np.linspace(self.rmin, self.rmax, self.Nr, dtype=np.float64)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz, dtype=np.float64)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')

        # Volume weights
        self.volume_field = self.compute_volume_field()

        # Anode (bool mask)
        self.rmin_anode = float(rmin_anode)
        self.rmax_anode = self.rmax
        self.zmin_anode = float(zmin_anode)
        self.zmax_anode = float(zmax_anode)
        self.zmin_anode2 = float(zmin_anode2)
        
        self.anode1_mask = (
            (self.r[:, None] >= self.rmin_anode) & (self.r[:, None] <= self.rmax_anode) &
            (self.z[None, :] >= self.zmin_anode) & (self.z[None, :] <= self.zmax_anode)
        )

        self.anode2_mask = (
            (self.r[:, None] >= self.rmin_anode) & (self.r[:, None] <= self.rmax_anode) &
            (self.z[None, :] >= self.zmin_anode2) & (self.z[None, :] <= self.zmin_anode2+(self.zmax_anode-self.zmin_anode))
        )

        # Cathode (bool mask)
        self.rmin_cathode = 0.0
        self.rmax_cathode = float(rmax_cathode)
        self.zmin_cathode = 0.0
        self.zmax_cathode = float(zmax_cathode)
        self.cathode_mask = (
            (self.r[:, None] >= self.rmin_cathode) & (self.r[:, None] <= self.rmax_cathode) &
            (self.z[None, :] >= self.zmin_cathode) & (self.z[None, :] <= self.zmax_cathode)
        )

        # Solve mask: int8 (1 = solve, 0 = masked)
        self.mask = np.ones((self.Nr, self.Nz), dtype=np.int8)
        self.mask[self.cathode_mask] = 0
        self.mask[self.anode1_mask] = 0
        self.mask[self.anode2_mask] = 0

        # Cathode and anode temperatures
        self.temperature_cathode = temperature_cathode
        self.temperature_anode = temperature_anode

        # Coils store (outside-domain only)
        self.coils: Dict[str, Dict[str, float]] = {}

    @property
    def n_coils(self) -> int:
        return len(self.coils)

    # ---------- Coil helpers (outside-domain enforcement) ----------

    def _intervals_overlap_closed(self, a0: float, a1: float, b0: float, b1: float) -> bool:
        """Return True if [a0,a1] overlaps (even at a point) with [b0,b1]."""
        # Ensure ordering
        if a0 > a1: a0, a1 = a1, a0
        if b0 > b1: b0, b1 = b1, b0
        return max(a0, b0) <= min(a1, b1)

    def _assert_coil_outside_domain(self, rc: float, drc: float, zc: float, dzc: float) -> None:
        """
        Raise if the coil window intersects or touches the simulation domain.
        """
        if drc <= 0 or dzc <= 0:
            raise ValueError("drc and dzc must be positive.")
        if rc < 0:
            raise ValueError("rc must be >= 0 for cylindrical geometry.")

        r0, r1 = rc - 0.5 * drc, rc + 0.5 * drc
        z0, z1 = zc - 0.5 * dzc, zc + 0.5 * dzc

        r_overlap = self._intervals_overlap_closed(r0, r1, self.rmin, self.rmax)
        z_overlap = self._intervals_overlap_closed(z0, z1, self.zmin, self.zmax)

        if r_overlap and z_overlap:
            # Overlap in both r and z => coil rectangle intersects/touches domain
            raise ValueError(
                "Coil window intersects the simulation domain. "
                "Biot-Savart solver requires coils to be entirely outside the domain. "
                f"Coil r∈[{r0:.6g},{r1:.6g}], z∈[{z0:.6g},{z1:.6g}] vs "
                f"domain r∈[{self.rmin:.6g},{self.rmax:.6g}], z∈[{self.zmin:.6g},{self.zmax:.6g}]."
            )

    def coil_overlaps_domain(self, name: str) -> bool:
        """Check if a named coil overlaps/touches the domain."""
        c = self.get_coil(name)
        r0, r1 = c["rc"] - 0.5 * c["drc"], c["rc"] + 0.5 * c["drc"]
        z0, z1 = c["zc"] - 0.5 * c["dzc"], c["zc"] + 0.5 * c["dzc"]
        r_overlap = self._intervals_overlap_closed(r0, r1, self.rmin, self.rmax)
        z_overlap = self._intervals_overlap_closed(z0, z1, self.zmin, self.zmax)
        return r_overlap and z_overlap

    def validate_coils_outside_domain(self) -> None:
        """Raise if any stored coil overlaps/touches the domain."""
        offenders = [n for n in self.coils if self.coil_overlaps_domain(n)]
        if offenders:
            raise ValueError(f"These coils overlap the domain: {offenders}")

    # ----------------------- Coil API -----------------------

    def add_coil(
        self,
        name: str,
        rc: float,
        drc: float,
        zc: float,
        dzc: float,
        current: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        """
        Add a rectangular coil centered at (rc, zc) with extents (drc, dzc).
        Coils MUST be strictly outside the simulation domain (no touching).
        """
        if not overwrite and name in self.coils:
            raise KeyError(f"Coil '{name}' already exists. Use overwrite=True to replace.")

        # Enforce 'outside-domain' constraint
        self._assert_coil_outside_domain(rc, drc, zc, dzc)

        self.coils[name] = {
            "rc": float(rc),
            "drc": float(drc),
            "zc": float(zc),
            "dzc": float(dzc),
            "current": float(current),
        }

    def set_coil_current(self, name: str, current: float) -> None:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        self.coils[name]["current"] = float(current)

    def get_coil(self, name: str) -> Dict[str, float]:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        return dict(self.coils[name])

    def remove_coil(self, name: str) -> None:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        del self.coils[name]

    # ---------------- Volume field ----------------

    def compute_volume_field(self) -> np.ndarray:
        """
        Axisymmetric control-volume weights (Verboncoeur-style corrections).
        Returns shape (Nr, Nz).
        """
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz

        volume_r = np.empty((Nr,), dtype=np.float64)

        # Axis node
        volume_r[0] = (np.pi / 3.0) * dr * dr * dz

        if Nr > 1:
            i_vals = np.arange(1, Nr - 1, dtype=np.float64)  # 1..Nr-2
            volume_r[1:Nr - 1] = 2.0 * np.pi * i_vals * dr * dr * dz

        # Outer ring node
        volume_r[Nr - 1] = np.pi * ((Nr - 1) - 1.0 / 3.0) * dr * dr * dz

        return np.repeat(volume_r[:, None], Nz, axis=1)
    

    def build_fem_mesh(
        self,
        characteristic_length_min: float | None = None,
        characteristic_length_max: float | None = None,
        refine_near_curves: bool = True,
    ):
        """
        Build a GMsh OCC geometry that matches this Geometry's domain and electrodes,
        then convert to a DOLFINx mesh and store it on `self.fem`.

        Geometry used (x=r, y=z):
        - Outer box: r ∈ [0, rmax], z ∈ [zmin, zmax]
        - Cathode slot removed:   [0, rmax_cathode] x [zmin, zmin+zmax_cathode]
        - TWO anode notches removed, both reaching r = rmax:
                A1: [rmin_anode, rmax] x [zmin_anode, zmax_anode]
                A2: [rmin_anode, rmax] x [zmin_anode2, zmin_anode2+(zmax_anode-zmin_anode)]
            Dirichlet is applied on the *internal anode island edges*,
            and on the right outer wall:
                z ∈ [zmax_anode, zmin_anode2]  (RIGHT_MID)  and
                z ∈ [zmin_anode2+(zmax_anode-zmin_anode), zmax] (RIGHT_TOP).
            The right wall below the first anode (z < zmin_anode) is natural (no-flux).

        Stores on self:
            self.fem.mesh        : dolfinx.mesh.Mesh
            self.fem.cell_tags   : dolfinx.mesh.MeshTags (cells)
            self.fem.facet_tags  : dolfinx.mesh.MeshTags (facets)
            self.fem.tags        : dict of tag integers (e.g. 'CATH_TOP', ...)
            self.fem.dx, self.fem.ds : UFL measures bound to the mesh/tags

        Returns:
            self.fem  (a SimpleNamespace with the fields above)
        """
        # Imports local to avoid hard dependency at module import time
        from types import SimpleNamespace
        from mpi4py import MPI
        from dolfinx.io import gmshio
        from dolfinx import fem as _fem
        import numpy as _np

        comm = MPI.COMM_WORLD
        rank = comm.rank

        # --- Sanity and derived anode-2 extent ---
        anode_height = float(self.zmax_anode - self.zmin_anode)
        zmin_a2 = float(self.zmin_anode2)
        zmax_a2 = zmin_a2 + anode_height

        # --- Allow the caller to skip size args (pick reasonable defaults) ---
        lc_min = characteristic_length_min or min(self.rmax, (self.zmax - self.zmin)) / 200.0
        lc_max = characteristic_length_max or min(self.rmax, (self.zmax - self.zmin)) / 60.0

        # ---------------- GMsh on rank 0 only ----------------
        model = None
        if rank == 0:
            try:
                import gmsh
            except Exception as exc:
                raise ImportError("gmsh is required to build the FEM mesh.") from exc

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc_min)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_max)

            gmsh.model.add("axisym_occ")
            occ = gmsh.model.occ

            # --- Primitives (use absolute z with self.zmin offset) ---
            outer = occ.addRectangle(0.0, self.zmin, 0.0, self.rmax, self.zmax - self.zmin)
            cath  = occ.addRectangle(
                0.0, self.zmin, 0.0,
                self.rmax_cathode,
                (self.zmax_cathode - self.zmin_cathode)  # typically zmin_cathode==0
            )

            anod1 = occ.addRectangle(
                self.rmin_anode, self.zmin_anode, 0.0,
                self.rmax - self.rmin_anode,
                self.zmax_anode - self.zmin_anode
            )
            anod2 = occ.addRectangle(
                self.rmin_anode, zmin_a2, 0.0,
                self.rmax - self.rmin_anode,
                zmax_a2 - zmin_a2
            )

            # --- Boolean cut: plasma = outer \ (cath ∪ anod1 ∪ anod2) ---
            cut = occ.cut([(2, outer)], [(2, cath), (2, anod1), (2, anod2)],
                        removeObject=True, removeTool=True)
            occ.synchronize()

            plasma_surfaces = [e for e in cut[0] if e[0] == 2]
            if not plasma_surfaces:
                gmsh.write("debug_no_surface.brep")
                gmsh.finalize()
                raise RuntimeError("No plasma surface after boolean cut (wrote debug_no_surface.brep).")

            gmsh.model.addPhysicalGroup(2, [s[1] for s in plasma_surfaces], G_DOMAIN)

            # --- Helper: bbox ---
            def bbox(dim, tag):
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, tag)
                return xmin, ymin, xmax, ymax

            tol = 1e-6
            cath_top_y = self.zmin + (self.zmax_cathode - self.zmin_cathode)  # top of slot
            top_all = max(self.zmax_anode, zmax_a2)

            # Collect all plasma boundary curves
            bnd = gmsh.model.getBoundary(plasma_surfaces, oriented=False, recursive=False)
            curve_tags = sorted({tag for (dim, tag) in bnd if dim == 1})

            cath_top_curves   = []
            anode_edge_curves = []
            right_mid_curves  = []
            right_top_curves  = []

            for tag in curve_tags:
                xmin, ymin, xmax, ymax = bbox(1, tag)

                # Cathode top: horizontal at y=cath_top_y for r∈[0,rmax_cathode]
                if abs(ymin - cath_top_y) < tol and abs(ymax - cath_top_y) < tol \
                and (xmax >= 0.0 - tol) and (xmin <= self.rmax_cathode + tol):
                    cath_top_curves.append(tag)
                    continue

                # Internal anode edges (for both notches):
                is_inner_vertical = (abs(xmin - self.rmin_anode) < tol and abs(xmax - self.rmin_anode) < tol)
                spans_a1 = (ymax >= self.zmin_anode - tol) and (ymin <= self.zmax_anode + tol)
                spans_a2 = (ymax >= zmin_a2          - tol) and (ymin <= zmax_a2          + tol)
                if is_inner_vertical and (spans_a1 or spans_a2):
                    anode_edge_curves.append(tag);  continue

                is_horiz_a1 = ((abs(ymin - self.zmin_anode) < tol and abs(ymax - self.zmin_anode) < tol) or
                            (abs(ymin - self.zmax_anode) < tol and abs(ymax - self.zmax_anode) < tol))
                is_horiz_a2 = ((abs(ymin - zmin_a2) < tol and abs(ymax - zmin_a2) < tol) or
                            (abs(ymin - zmax_a2) < tol and abs(ymax - zmax_a2) < tol))
                if (is_horiz_a1 or is_horiz_a2) and (xmin >= self.rmin_anode - tol):
                    anode_edge_curves.append(tag);  continue

                # Right wall at r≈R: split into three intervals
                on_right_wall = (abs(xmin - self.rmax) < tol and abs(xmax - self.rmax) < tol)
                if on_right_wall:
                    # bottom [zmin, zmin_anode): natural
                    # mid    [zmax_anode, zmin_anode2]: Dirichlet (RIGHT_MID)
                    # top    [zmax_a2, zmax]:          Dirichlet (RIGHT_TOP)
                    if ymax <= self.zmin_anode + tol:
                        continue
                    elif ymin >= zmax_a2 - tol:
                        right_top_curves.append(tag);  continue
                    elif (ymax >= self.zmax_anode - tol) and (ymin <= zmin_a2 + tol):
                        right_mid_curves.append(tag);  continue
                    # Rare straddle: use center
                    ymid = 0.5*(ymin+ymax)
                    if (self.zmax_anode - tol) <= ymid <= (zmin_a2 + tol):
                        right_mid_curves.append(tag)
                    elif ymid >= zmax_a2 - tol:
                        right_top_curves.append(tag)
                    continue

            # Register facet groups (assert presence)
            if not cath_top_curves:
                gmsh.write("debug_no_cath_top.brep")
                gmsh.finalize()
                raise RuntimeError("No cathode-top curve found (wrote debug_no_cath_top.brep).")
            gmsh.model.addPhysicalGroup(1, cath_top_curves, G_CATH_TOP)

            if not anode_edge_curves:
                gmsh.write("debug_no_anode_edges.brep")
                gmsh.finalize()
                raise RuntimeError("No anode internal edges found (wrote debug_no_anode_edges.brep).")
            gmsh.model.addPhysicalGroup(1, anode_edge_curves, G_ANODE_ISL)

            if not right_mid_curves:
                gmsh.write("debug_no_right_mid.brep")
                gmsh.finalize()
                raise RuntimeError("No right-wall BETWEEN-anodes segment found (wrote debug_no_right_mid.brep).")
            gmsh.model.addPhysicalGroup(1, right_mid_curves, G_RIGHT_MID)

            if not right_top_curves:
                gmsh.write("debug_no_right_top.brep")
                gmsh.finalize()
                raise RuntimeError("No right-wall ABOVE-anodes segment found (wrote debug_no_right_top.brep).")
            gmsh.model.addPhysicalGroup(1, right_top_curves, G_RIGHT_TOP)

            # Optional refinement near sensitive edges
            if refine_near_curves and (anode_edge_curves or cath_top_curves):
                f_dist = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", anode_edge_curves + cath_top_curves)
                gmsh.model.mesh.field.setNumber(f_dist, "NumPointsPerCurve", 200)
                f_th = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
                gmsh.model.mesh.field.setNumber(f_th, "SizeMin", lc_min/2)
                gmsh.model.mesh.field.setNumber(f_th, "SizeMax", lc_max)
                gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.003)
                gmsh.model.mesh.field.setNumber(f_th, "DistMax", 0.01)
                gmsh.model.mesh.field.setAsBackgroundMesh(f_th)

            gmsh.model.mesh.generate(2)
            model = gmsh.model

        # --- Convert to DOLFINx on all ranks ---
        msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, 0, gdim=2)

        # Only rank 0 had gmsh; finalize there
        if rank == 0:
            import gmsh  # type: ignore
            gmsh.finalize()

        # Ensure connectivity needed for locate_dofs_topological, etc.
        tdim = msh.topology.dim
        msh.topology.create_connectivity(tdim - 1, tdim)
        msh.topology.create_connectivity(tdim - 1, 0)

        # Build measures
        dx = _fem.measure.dx(domain=msh)
        ds = _fem.measure.ds(domain=msh, subdomain_data=facet_tags)

        # Stash everything on self.fem
        tags = {
            "DOMAIN":    G_DOMAIN,
            "CATH_TOP":  G_CATH_TOP,
            "ANODE_ISL": G_ANODE_ISL,
            "RIGHT_MID": G_RIGHT_MID,
            "RIGHT_TOP": G_RIGHT_TOP,
        }
        self.fem = SimpleNamespace(
            mesh=msh,
            cell_tags=cell_tags,
            facet_tags=facet_tags,
            tags=tags,
            dx=dx,
            ds=ds,
        )

        # A small summary on rank 0
        if rank == 0:
            n_nodes = self.fem.mesh.geometry.x.shape[0]
            n_cells = self.fem.mesh.topology.index_map(self.fem.mesh.topology.dim).size_local
            print(f"[Geometry] FEM mesh: {n_nodes} nodes, {n_cells} cells")
            for k, gid in tags.items():
                if k == "DOMAIN":
                    continue
                print(f"[Geometry] Facets[{k}] = {self.fem.facet_tags.find(gid).size}")

        return self.fem