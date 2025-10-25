import numpy as np

from centrifugesim.fluids import neutral_fluid_kernels_numba
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class NeutralFluidContainer:
    """
    """
    def __init__(self, geom:Geometry, nn_floor, mass):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.nn_floor = nn_floor
        self.mass = mass

        self.nn_grid = np.zeros((self.Nr, self.Nz))

        self.un_r_grid = np.zeros((self.Nr, self.Nz))
        self.un_theta_grid = np.zeros((self.Nr, self.Nz))
        self.un_z_grid = np.zeros((self.Nr, self.Nz))
        
        self.T_n_grid = np.zeros((self.Nr, self.Nz))