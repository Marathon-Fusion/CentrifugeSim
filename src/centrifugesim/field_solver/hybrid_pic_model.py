import numpy as np
import cupy as cp

from centrifugesim.geometry.geometry import Geometry
from centrifugesim.field_solver.fem_phi_solver import (
    init_phi_coeffs, update_phi_coeffs_from_grids, solve_phi_axisym,
    functions_to_rect_grids, _get_rect_sampler
)
from centrifugesim import constants


class HybridPICModel:
    def __init__(self, geom:Geometry):
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

        # ------- Device fields ---------
        self.Er_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Et_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Ez_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

        self.Br_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bt_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bz_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

        # --- Precreate FEM coefficient functions
        self.coeffs = init_phi_coeffs(geom)

        # --- Precreate a rect-grid sampler (once). Reuse across steps ---
        self.sampler = _get_rect_sampler(geom, Nr=None, Nz=None)  # defaults to geom.Nr x geom.Nz

        # applied current and solution current
        self.I_app = 0
        self.I_sol = 0

        self.sol = {}

    def compute_B_aux(self):
        self.Bmag_grid[:] = np.sqrt(self.Br_grid**2 + self.Bz_grid**2)
        self.br_grid[:] = np.where(self.Bmag_grid==0, 0, self.Br_grid/self.Bmag_grid)
        self.bz_grid[:] = np.where(self.Bmag_grid==0, 0, self.Bz_grid/self.Bmag_grid)

        self.Br_grid_d = cp.asarray(self.Br_grid)
        self.Bz_grid_d = cp.asarray(self.Bz_grid)

    def update_phi_coeffs(self, geom:Geometry, electron_fluid, neutral_fluid):
        update_phi_coeffs_from_grids(
            geom, self.coeffs,
            ne_grid=electron_fluid.ne_grid,
            grad_pe_grid_r=electron_fluid.grad_pe_grid_r,
            grad_pe_grid_z=electron_fluid.grad_pe_grid_z,
            sigma_parallel_grid=electron_fluid.sigma_parallel_grid,
            sigma_P_grid=electron_fluid.sigma_P_grid,
            sigma_H_grid=electron_fluid.sigma_H_grid,
            Bz_grid=self.Bz_grid,
            un_r_grid=neutral_fluid.un_r_grid,
            un_theta_grid=neutral_fluid.un_theta_grid,
        )

    def solve_phi_and_update_fields_grid(self, geom:Geometry, Jz0, sigma_r, initial_solve=True):
        """
        Need to run update_phi_coeffs first
        """
        if(initial_solve):
            sol = solve_phi_axisym(geom, self.coeffs, Jz0=Jz0, sigma_r=sigma_r, phi_a=0.0)
        else:
            sol = solve_phi_axisym(geom, self.coeffs, Jz0=Jz0, sigma_r=sigma_r, phi_a=0.0,
                phi_guess=self.sol["phi"])

        fields = {k: sol[k] for k in ("ne", "phi", "Er", "Ez", "Jr", "Jz", "q_ohm")}
        rect = functions_to_rect_grids(geom, fields, sampler=self.sampler)

        self.phi_grid = rect["phi"]
        self.Jer_grid = rect["Jr"]; self.Jez_grid = rect["Jz"]
        self.Er_grid = rect["Er"]; self.Ez_grid = rect["Ez"]
        self.q_ohm_grid = rect["q_ohm"]

        # setting masked cathode and anode values to 0
        self.phi_grid[np.isnan(self.phi_grid)] = 0
        self.Jer_grid[np.isnan(self.Jer_grid)] = 0
        self.Jez_grid[np.isnan(self.Jez_grid)] = 0
        self.Er_grid[np.isnan(self.Er_grid)] = 0
        self.Ez_grid[np.isnan(self.Ez_grid)] = 0
        self.q_ohm_grid[np.isnan(self.q_ohm_grid)] = 0

        # ensure proper axis behavior
        self.Er_grid[0,:] = 0
        self.Jer_grid[0,:] = 0

        self.Ez_grid[0,:] = self.Ez_grid[1,:]
        self.Jez_grid[0,:] = self.Jez_grid[1,:]

        # copy E field components to gpu
        self.Er_grid_d = cp.asarray(self.Er_grid)
        self.Ez_grid_d = cp.asarray(self.Ez_grid)

        self.I_app = sol["integrals"]["I_applied"]
        self.I_sol = sol["integrals"]["I_from_solution"]

        self.sol = sol