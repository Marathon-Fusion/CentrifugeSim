import numpy as np

from centrifugesim.fluids import electron_fluid_kernels_numba
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class ElectronFluidContainer:
    def __init__(self, geom:Geometry):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.ne = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !
        self.pe = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa] !

        self.kappa_parallel = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.kappa_perp = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.nu_ei = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_en = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

    def initialize_Te(self):
        None

    def update_kappa(self):
        """
        Add functions here for Braginskii or other as needed
        """
        self.kappa_parallel[:,:] = 0
        self.kappa_perp[:,:] = 0

    def update_nu(self):
        """
        Add functions here to update collision frequencies
        """
        self.nu_ei[:,:] = 0
        self.nu_en[:,:] = 0

    def update_sigma(self):
        """
        Add functions here to update conductivities (Pedersen, Hall, parallel)
        Add other if necessary, check Ohms law solver.
        """
        None

    def update_Te(self, geometry, n_n, Br, Bz, T_i, T_n, m_ion, m_neutral, dt):
        "Update Te function solving energy equation"
        ni = self.ne
        Te_new = np.zeros_like(self.Te)

        # Complete these functions (arguments, etc)
        #Q_J = self.compute_Joule_heating()
        #self.update_nu(...) 

        # Should move this to geometry or hybrid_pic to avoid calculating it multiple times
        B_mag = np.sqrt(Br**2 + Bz**2) + 1e-12
        br = Br / B_mag
        bz = Bz / B_mag
        # ------------------------------------------

        Te_new = electron_fluid_kernels_numba.solve_step(self.Te, Te_new, geometry.dr, geometry.dz, geometry.r_vec, self.ne, ni, n_n, Q_J,
                                                        br, bz, B_mag, T_i, T_n, self.kappa_parallel, self.kappa_perp,
                                                        self.nu_ei, self.nu_en, geometry.mask, m_ion, m_neutral, geometry.Nr, geometry.Nz, dt)
        self.Te[:,:] = np.copy(Te_new)

    def compute_Joule_heating(self):
        
        Q_J = np.zeros_like(self.Te)
        
        return Q_J

    def apply_boundary_conditions(self):
        """
        Applies Neumann (zero-flux) boundary conditions on the outer computational walls.

        Need to revisit this function !!! 
        Currently masked regions act as dirichlet Boundary conditions !
        Should move physical walls boundaries to Dirichlet BCs
        """
        # Axis of symmetry (r=0): dTe/dr = 0
        self.Te[0, :] = self.Te[1, :]

        # Outer walls: zero-flux
        self.Te[-1, :] = self.Te[-2, :]
        self.Te[:, 0] = self.Te[:, 1]
        self.Te[:, -1] = self.Te[:, -2]