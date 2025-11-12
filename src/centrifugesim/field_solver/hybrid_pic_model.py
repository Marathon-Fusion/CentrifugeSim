import numpy as np
import cupy as cp

from centrifugesim.geometry.geometry import Geometry
from centrifugesim.field_solver.fem_phi_solver import (
    init_phi_coeffs, update_phi_coeffs_from_grids, solve_phi_axisym,
    functions_to_rect_grids, _get_rect_sampler
)
from centrifugesim.field_solver.finite_volume_phi_solver import solve_anisotropic_poisson_FV, compute_E_and_J
from centrifugesim import constants


class HybridPICModel:
    def __init__(self, geom:Geometry, use_fem = False):
        # geometry info
        self.zmin = geom.zmin
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        self.dr = geom.dr
        self.dz = geom.dz
        self.r  = geom.r   # 1D array of length Nr (cell centers)
        self.z  = geom.z

        # for cathode boundary condition
        self.dphi_dz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)
        self.Jz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)

        # fields
        self.phi_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Er_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Et_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)  # unused in solver; kept for pusher
        self.Ez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # electron current density components
        self.Jer_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # q_ohm for electron energy equation
        self.q_ohm_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Br_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Bt_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)  # unused in solver; kept for pusher
        self.Bz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.br_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64) # Br/Bmag
        self.bz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64) # Bz/Bmag
        self.Bmag_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # ------- Device fields ---------
        self.Er_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Et_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Ez_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

        self.Br_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bt_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bz_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

        if(use_fem):
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


    def compute_dphi_dz_cathode(self, geom:Geometry, I, electron_fluid, rmax_injection=None, sigma_r=None):
        # I is negative (enters cathode)

        # Should be input instead
        # add here check that sigma_r is smaller than rmax_injection/2, ideally /3
        rmax_injection = geom.rmax_cathode
        sigma_r = rmax_injection/3.0

        dphi_dz_vec = np.zeros(self.Nr).astype(np.float64)

        i_cathode = (np.arange(self.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)

        # calculate input current density (Jz0 is negative)
        Jz0 = I / (2*np.pi*sigma_r**2)
        Jz_cathode = Jz0*np.exp(-0.5*geom.r[i_cathode]**2 / sigma_r**2)

        sigma_parallel_cathode = electron_fluid.sigma_parallel_grid[i_cathode, j_cathode]

        # dphi_dz = (Jiz-Jz)/sigma_parallel + dpe/dz /(e*ne) at cathode
        # but using only Jz for now to test
        dphi_dz_vec_aux = -Jz_cathode/sigma_parallel_cathode # change to dphi_dz = (Jiz-Jz)/sigma_parallel + dpe/dz /(e*ne) at cathode

        dphi_dz_vec[i_cathode] = - dphi_dz_vec_aux # flipping sign here due to how the solver was written

        self.dphi_dz_cathode_top_vec = np.copy(dphi_dz_vec)
        self.Jz_cathode_top_vec[i_cathode] = np.copy(Jz_cathode)

    #-----------------------------------------------------------------------------
    #------------------------------ Calls to FV solver ---------------------------
    #-----------------------------------------------------------------------------
    def solve_phi_and_update_fields_grid_FV(self,
        geom:Geometry,
        electron_fluid,
        neutral_fluid,
        tol=1e-9, max_iter=100_000,
        phi_anode_value=0,
        Ji_r=None, Ji_z=None,
        phi0=None, verbose=True):

        phi, info = solve_anisotropic_poisson_FV(
            geom,
            electron_fluid.sigma_P_grid,
            electron_fluid.sigma_parallel_grid,
            ne=electron_fluid.ne_grid,
            pe=electron_fluid.pe_grid,
            Bz=self.Bz_grid,
            un_theta=neutral_fluid.un_theta_grid,
            ne_floor=electron_fluid.ne_floor,
            Ji_r=Ji_r, Ji_z=Ji_z,
            dphi_dz_cathode_top=self.dphi_dz_cathode_top_vec,
            phi_anode_value=phi_anode_value,
            phi0=phi0,
            omega=1.8, tol=tol, max_iter=max_iter,
            verbose=verbose
        )

        Er, Ez, Jer, Jez = compute_E_and_J(phi, geom,
                    electron_fluid.sigma_P_grid,
                    electron_fluid.sigma_parallel_grid,
                    ne=electron_fluid.ne_grid,
                    pe=electron_fluid.pe_grid,
                    Bz=self.Bz_grid,
                    un_theta=neutral_fluid.un_theta_grid,
                    ne_floor=electron_fluid.ne_floor,
                    fill_solid_with_nan=False)

        # TO DO: move to kernel
        q_ohm = electron_fluid.sigma_P_grid*Er*Er + electron_fluid.sigma_parallel_grid*Ez*Ez

        self.phi_grid = np.copy(phi)
        self.Er_grid = np.copy(Er)
        self.Ez_grid = np.copy(Ez)
        self.Jer_grid = np.copy(Jer)
        self.Jez_grid = np.copy(Jez)
        self.q_ohm_grid = np.copy(q_ohm)

        del phi, Er, Ez, Jer, Jez, q_ohm

    #-----------------------------------------------------------------------------
    #----------------------------- Calls to FEM solver ---------------------------
    #-----------------------------------------------------------------------------
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