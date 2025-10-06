import numpy as np

class Geometry:
    """
    Represents a 2D RZ grid geometry with separate boundary conditions
    and custom particle boundary surfaces on each side.
    """
    def __init__(
        self,
        rmax: float,
        zmin: float,
        zmax: float,
        Nr: int,
        Nz: int,
        #rmax_particles: np.ndarray,
    ):
        """
        Initialize geometry parameters and compute grid arrays.

        Parameters:
        -----------
        ...
        """
        # Domain limits
        self.rmin = 0 # hardcoded for now
        self.rmax = float(rmax)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.Nr = int(Nr)
        self.Nz = int(Nz)

        # Grid spacings
        self.dr = (self.rmax - self.rmin) / (self.Nr - 1)
        self.dz = (self.zmax - self.zmin) / (self.Nz - 1)

        # Node-centered coordinate arrays
        self.r = np.linspace(self.rmin, self.rmax, self.Nr, dtype=np.float64)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz, dtype=np.float64)

        # Create 2D fields using meshgrid in row-major order.
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')

        # Particle boundary surfaces
        # rmax_particles: user-specified array of length nz
        #if not isinstance(rmax_particles, np.ndarray) or rmax_particles.shape != (Nz,):
        #    raise ValueError("rmax_particles must be a numpy array of length nz.")
        #self.rmax_particles = rmax_particles.astype(np.float64)