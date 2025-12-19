import numpy as np
from centrifugesim import constants
import ion_fluid_helper

class IonFluidContainer:
    def __init__(self, geom, m_i, Z=1.0, sigma_cx=5e-19, Ti0=300.0):
        """
        Container for Ion Fluid moments and transport coefficients.
        
        Args:
            geom: Geometry object containing .Nr, .Nz, .mask
            m_i: Ion mass in kg
            Z: Ion charge state
            sigma_cx: Charge Exchange cross-section (m^2)
            Ti0: Initial ion temperature (K)
        """
        self.geom = geom
        self.m_i = float(m_i)
        self.Z = float(Z)
        self.sigma_cx = sigma_cx
        
        # Grid Dimensions
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        shape = (self.Nr, self.Nz)
        
        # --- Primary Fields ---
        self.vtheta = np.zeros(shape, dtype=np.float64)
        self.ni_grid = np.zeros(shape, dtype=np.float64)
        self.Ti_grid = Ti0 + np.zeros(shape, dtype=np.float64)
        
        # --- Collision & Magnetization ---
        self.nu_i_grid = np.zeros(shape, dtype=np.float64)   # Total collision freq (s^-1)
        self.beta_i_grid = np.zeros(shape, dtype=np.float64) # Hall parameter (wci/nu_i)
        
        # --- Transport Coefficients ---
        self.eta_0 = np.zeros(shape, dtype=np.float64)               # Parallel Viscosity
        self.sigma_P_grid = np.zeros(shape, dtype=np.float64)        # Pedersen Conductivity
        self.sigma_parallel_grid = np.zeros(shape, dtype=np.float64) # Parallel Conductivity

    def update_vtheta(self, geom, hybrid_pic, nu_in_grid, neutral_fluid):
        """
        Updates self.vtheta using the algebraic approximation (Drag = JxB).
        """
        # Ensure density is synced
        if hasattr(hybrid_pic, 'ni_grid'):
            self.ni_grid[:] = hybrid_pic.ni_grid

        ion_fluid_helper.update_vtheta_kernel_algebraic(
            self.vtheta,                  # Output
            hybrid_pic.Jer_grid,          # Input
            hybrid_pic.Bz_grid,           # Input
            self.ni_grid,                 # Input
            nu_in_grid,                   # Input
            neutral_fluid.un_theta_grid,  # Input
            geom.mask,                    # Geometry
            self.m_i                      # Constant
        )

    def update_collision_frequencies(self, geom, electron_fluid, neutral_fluid, hybrid_pic):
        """
        Calculates nu_i = nu_ii (Coulomb) + nu_in (Charge Exchange).
        """
        # Ensure density is synced
        if hasattr(hybrid_pic, 'ni_grid'):
            self.ni_grid[:] = hybrid_pic.ni_grid
            
        ion_fluid_helper.compute_nu_i_kernel(
            self.nu_i_grid,    # Output
            self.ni_grid,      # Input
            self.Ti_grid,      # Input
            neutral_fluid.nn_grid,     # Input
            neutral_fluid.T_n_grid,    # Input
            self.Z,
            self.m_i,
            self.sigma_cx,
            geom.mask,
            constants.eps_0,
            constants.q_e,
            constants.kb
        )

    def update_beta_i(self, geom, hybrid_pic):
        """
        Calculates beta_i = wci / nu_i
        """
        ion_fluid_helper.compute_beta_i_kernel(
            self.beta_i_grid,   # Output
            self.nu_i_grid,     # Input (must be updated first)
            hybrid_pic.Bz_grid, # Input
            self.Z,
            constants.q_e,
            self.m_i,
            geom.mask
        )

    def update_conductivities(self, geom):
        """
        Calculates sigma_P and sigma_parallel.
        """
        ion_fluid_helper.compute_conductivities_kernel(
            self.sigma_P_grid,        # Output
            self.sigma_parallel_grid, # Output
            self.ni_grid,
            self.nu_i_grid,
            self.beta_i_grid,
            self.Z,
            constants.q_e,
            self.m_i,
            geom.mask
        )

    def compute_viscosity_eta0(self, geom):
        """
        Computes parallel viscosity eta_0.
        """
        self.eta_0[geom.mask==1] = 0.96 * self.ni_grid[geom.mask==1] * constants.kb * self.Ti_grid[geom.mask==1] / self.nu_i_grid[geom.mask==1]