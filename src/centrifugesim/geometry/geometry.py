import numpy as np
from typing import Dict

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
        self.anode_mask = (
            (self.r[:, None] >= self.rmin_anode) & (self.r[:, None] <= self.rmax_anode) &
            (self.z[None, :] >= self.zmin_anode) & (self.z[None, :] <= self.zmax_anode)
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
        self.mask[self.anode_mask] = 0

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
                "Biot–Savart solver requires coils to be entirely outside the domain. "
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