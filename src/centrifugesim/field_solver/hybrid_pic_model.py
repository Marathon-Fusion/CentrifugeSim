import numpy as np

#from centrifugesim.fluids import electron_fluid_kernels_numba

from centrifugesim.field_solver.fem_phi_solver import (
    init_phi_coeffs, update_phi_coeffs_from_grids, solve_phi_axisym,
    functions_to_rect_grids, _get_rect_sampler
)
from centrifugesim import constants


class HybridPICModel:
    def __init__(self, geom):
        # geometry info
        self.zmin = geom.zmin
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        self.dr = geom.dr
        self.dz = geom.dz
        self.r  = geom.r   # 1D array of length Nr (cell centers)
        self.z  = geom.z

        # fields
        self.phi_grid = np.zeros((self.Nr, self.Nz))

        self.Er_grid = np.zeros((self.Nr, self.Nz))
        self.Et_grid = np.zeros((self.Nr, self.Nz))  # unused in solver; kept for pusher
        self.Ez_grid = np.zeros((self.Nr, self.Nz))

        # electron current density components
        self.Jer_grid = np.zeros((self.Nr, self.Nz))
        self.Jez_grid = np.zeros((self.Nr, self.Nz))

        # q_ohm for electron energy equation
        self.q_ohm_grid = np.zeros((self.Nr, self.Nz))

        self.Br_grid = np.zeros((self.Nr, self.Nz))
        self.Bt_grid = np.zeros((self.Nr, self.Nz))  # unused in solver; kept for pusher
        self.Bz_grid = np.zeros((self.Nr, self.Nz))

        self.br_grid = np.zeros((self.Nr, self.Nz)) # Br/Bmag
        self.bz_grid = np.zeros((self.Nr, self.Nz)) # Bz/Bmag
        self.Bmag_grid = np.zeros((self.Nr, self.Nz))

        # Collision frequencies (placeholders)
        # Move to particle container?
        # self.nu_in_grid = np.zeros((self.Nr, self.Nz))
        # self.nu_cx_grid = np.zeros((self.Nr, self.Nz))

        # --- Precreate FEM coefficient functions
        self.coeffs = init_phi_coeffs(geom)

        # --- Precreate a rect-grid sampler (once). Reuse across steps ---
        self.sampler = _get_rect_sampler(geom, Nr=None, Nz=None)  # defaults to geom.Nr x geom.Nz

        # applied current and solution current
        self.I_app = 0
        self.I_sol = 0

        self.sol = {}

    def update_phi_coeffs(self, geom, electron_fluid, un_r_grid, un_theta_grid):
        update_phi_coeffs_from_grids(
            geom, self.coeffs,
            ne_grid=electron_fluid.ne_grid, Te_grid=electron_fluid.Te_grid,
            sigma_parallel_grid=electron_fluid.sigma_par_grid,
            sigma_P_grid=electron_fluid.sigma_P_grid,
            sigma_H_grid=electron_fluid.sigma_H_grid,
            Bz_grid=self.Bz_grid, un_r_grid=un_r_grid, un_theta_grid=un_theta_grid,
        )

    def solve_phi_and_sample_to_rect_grid(self, geom, Jz0, sigma_r, initial_solve=True):
        
        if(initial_solve):
            sol = solve_phi_axisym(geom, self.coeffs, Jz0=Jz0, sigma_r=sigma_r, phi_a=0.0)
        else:
            sol = solve_phi_axisym(geom, self.coeffs, Jz0=Jz0, sigma_r=sigma_r, phi_a=0.0,
                phi_guess=self.sol["phi"])

        fields = {k: sol[k] for k in ("ne", "phi", "Er", "Ez", "Jr", "Jz", "q_ohm")}
        rect = functions_to_rect_grids(geom, fields, sampler=self.sampler)

        self.phi_grid = rect["phi"]
        self.Jer_grid = rect["Jr_grid"]; self.Jez_grid = rect["Jz_grid"]
        self.Er_grid = rect["Er_grid"]; self.Ez_grid = rect["Ez_grid"]
        self.q_ohm_grid = rect["q_ohm"]

        self.ne_grid[np.isnan(self.ne_grid)] = 0
        self.Er_grid[np.isnan(self.Er_grid)] = 0
        self.Ez_grid[np.isnan(self.Ez_grid)] = 0
        self.q_ohm_grid[np.isnan(self.q_ohm_grid)] = 0

        self.I_app = sol["integrals"]["I_applied"]
        self.I_sol = sol["integrals"]["I_from_solution"]

        self.sol = sol