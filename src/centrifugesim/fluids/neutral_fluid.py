import numpy as np

from centrifugesim.fluids import neutral_fluid_kernels_numba
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class NeutralFluidContainer:
    """
    """
    def __init__(self, geom:Geometry, nn_floor, mass, name, Tn0=0.0):

        self.name = name

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.nn_floor = nn_floor
        self.mass = mass

        self.nn_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.un_r_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_theta_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_z_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        
        self.T_n_grid = Tn0*np.ones((self.Nr, self.Nz)).astype(np.float64)

        print(self.name + " initialized")

    def update_utheta(self, geom, ui_theta_grid, nu_in_grid, ni_grid, mi, dt):
        """
        Just to test before merging with compressible Navier Stokes equations.
        """
        dtnu_max = nu_in_grid[geom.mask==1].max()*dt
        if(dtnu_max>0.1):
            print("dt*nu_in.max() > 0.1 !", dtnu_max)

        un_theta_new = neutral_fluid_kernels_numba.test_update_u_theta(geom.mask,
                        ni_grid*mi, self.nn_grid*self.mass,
                        ui_theta_grid, self.un_theta,
                        nu_in_grid, self.nn_floor*self.mass, dt)

        self.un_theta[geom.mask==1] = np.copy(un_theta_new[geom.mask==1])