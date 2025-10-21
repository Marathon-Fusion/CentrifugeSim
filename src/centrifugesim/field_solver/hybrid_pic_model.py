import numpy as np

from centrifugesim.fluids import electron_fluid_kernels_numba
from centrifugesim import constants


class HybridPICModel:
    def __init__(self, geometry):
        # geometry info
        self.zmin = geometry.zmin
        self.Nr = geometry.Nr
        self.Nz = geometry.Nz
        self.dr = geometry.dr
        self.dz = geometry.dz
        self.r  = geometry.r   # 1D array of length Nr (cell centers)

        # fields
        self.phi = np.zeros((self.Nr, self.Nz))

        self.Er = np.zeros((self.Nr, self.Nz))
        self.Et = np.zeros((self.Nr, self.Nz))  # unused in solver; kept for pusher
        self.Ez = np.zeros((self.Nr, self.Nz))

        # electron current density components
        self.Jer = np.zeros((self.Nr, self.Nz))
        self.Jez = np.zeros((self.Nr, self.Nz))

        # q_ohm for electron energy equation
        self.q_ohm = np.zeros((self.Nr, self.Nz))

        self.Br = np.zeros((self.Nr, self.Nz))
        self.Bt = np.zeros((self.Nr, self.Nz))  # unused in solver; kept for pusher
        self.Bz = np.zeros((self.Nr, self.Nz))

        self.br = np.zeros((self.Nr, self.Nz)) # Br/Bmag
        self.bz = np.zeros((self.Nr, self.Nz)) # Bz/Bmag
        self.Bmag = np.zeros((self.Nr, self.Nz))

        # Electrical conductivities (tensor components)
        self.sigma_H = np.zeros((self.Nr, self.Nz))
        self.sigma_P = np.zeros((self.Nr, self.Nz))
        self.sigma_parallel = np.zeros((self.Nr, self.Nz))

        # Collision frequencies (placeholders)
        self.nu_in = np.zeros((self.Nr, self.Nz))
        self.nu_cx = np.zeros((self.Nr, self.Nz))

        self.nu_en = np.zeros((self.Nr, self.Nz))
        self.nu_ei = np.zeros((self.Nr, self.Nz))
        self.nu_e = np.zeros((self.Nr, self.Nz))


    # --- Add function to solve for phi ---
    """
    It should read geometry object as argument and internally convert ne, Te, Bz, sigma_H, sigma_parallel, sigma_P (self) numpy arrays to the fem.Function ones,
    then it should solve for phi using dolfinx, then calculate Er, Ez, Jr, Jz as well as q_ohm, convert those fields to numpy arrays and finally return the numpy
    arrays phi_grid, Er_grid, Ez_grid, Jr_grid, Jz_grid, q_ohm_grid. Write auxiliary methods of this class if needed to make the code clean.  
    """