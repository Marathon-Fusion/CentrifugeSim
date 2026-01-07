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

        self.t = 0 # current simulation time (s)

        # geometry info
        self.zmin = geom.zmin
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        self.dr = geom.dr
        self.dz = geom.dz
        self.r  = geom.r   # 1D array of length Nr (cell centers)
        self.z  = geom.z

        # for cathode boundary condition
        self.phi_cathode_vec = np.zeros(self.Nr).astype(np.float64)
        self.dphi_dz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)
        self.Jz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)

        # fields
        self.phi_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Er_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Et_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)  # unused in solver; kept for pusher
        self.Ez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Er_grid_grad_pe = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Ez_grid_grad_pe = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # Total conductivity components
        self.sigma_P_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.sigma_parallel_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # electron current density components
        self.Jer_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # ion current density components
        self.Jir_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jiz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # total current density components
        self.Jr_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # q_ohm for electron energy equation
        self.q_ohm_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_ohm_ions_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_ohm_electrons_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_RF = np.zeros((self.Nr, self.Nz)).astype(np.float64) # electrons only

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

    def update_conductivities(self, electron_fluid, ion_fluid):
        self.sigma_P_grid[:] = electron_fluid.sigma_P_grid + ion_fluid.sigma_P_grid
        self.sigma_parallel_grid[:] = electron_fluid.sigma_parallel_grid + ion_fluid.sigma_parallel_grid

    def update_Je_and_Ji_from_Jtotal(self, geom, electron_fluid, ion_fluid):
        """
        Updates self.Jer_grid, self.Jez_grid, self.Jir_grid, self.Jiz_grid
        from total current density Jr_grid, Jz_grid and components of conductivity.
        """
        self.Jer_grid*=0.0
        self.Jez_grid*=0.0
        self.Jir_grid*=0.0
        self.Jiz_grid*=0.0

        self.Jer_grid[geom.mask==1] = (electron_fluid.sigma_P_grid[geom.mask==1] / (electron_fluid.sigma_P_grid[geom.mask==1] + ion_fluid.sigma_P_grid[geom.mask==1])) * self.Jr_grid[geom.mask==1]
        self.Jez_grid[geom.mask==1] = (electron_fluid.sigma_parallel_grid[geom.mask==1] / (electron_fluid.sigma_parallel_grid[geom.mask==1] + ion_fluid.sigma_parallel_grid[geom.mask==1])) * self.Jz_grid[geom.mask==1]

        self.Jir_grid[geom.mask==1] = self.Jr_grid[geom.mask==1] - self.Jer_grid[geom.mask==1]
        self.Jiz_grid[geom.mask==1] = self.Jz_grid[geom.mask==1] - self.Jez_grid[geom.mask==1]
        
    def compute_dphi_dz_cathode(self, geom:Geometry, I, Jiz_grid=None, rmax_injection=None, sigma_r=None):
        # I is negative (enters cathode)

        # Should be input instead
        # add here check that sigma_r is smaller than rmax_injection/2, ideally /3
        rmax_injection = geom.rmax_cathode
        sigma_r = rmax_injection/3.0

        dphi_dz_vec = np.zeros(self.Nr).astype(np.float64)

        i_cathode = (np.arange(self.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)

        # calculate input current density (I is negative)
        Jz0 = I / (2*np.pi*sigma_r**2)
        Jz_cathode = Jz0*np.exp(-0.5*geom.r[i_cathode]**2 / sigma_r**2)

        sigma_parallel_cathode = self.sigma_parallel_grid[i_cathode, j_cathode]
        
        # dphi_dz = (Jiz-Jz)/sigma_parallel + dpe/dz /(e*ne) at cathode
        dphi_dz_vec_aux = -Jz_cathode/sigma_parallel_cathode
    
        if(Jiz_grid is not None):
            Jiz_cathode = Jiz_grid[i_cathode, j_cathode]
            dphi_dz_vec_aux += Jiz_cathode/sigma_parallel_cathode

        # commenting out for now to avoid unstable behavior at cathode face
        #dpe_dz_cathode = (electron_fluid.pe_grid[i_cathode, j_cathode+1] - electron_fluid.pe_grid[i_cathode, j_cathode])/geom.dz
        #ne_cathode = electron_fluid.ne_grid[i_cathode, j_cathode]
        #dphi_dz_vec_aux += dpe_dz_cathode/(constants.q_e*ne_cathode)
        
        # flipping sign here due to how the solver was written
        dphi_dz_vec[i_cathode] = - dphi_dz_vec_aux 

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
        phi0=None,
        cathode_dirichlet = False,
        verbose=True):

        if(cathode_dirichlet):
            phi, info = solve_anisotropic_poisson_FV(
                geom,
                self.sigma_P_grid,
                self.sigma_parallel_grid,
                ne=electron_fluid.ne_grid,
                pe=electron_fluid.pe_grid,
                Bz=self.Bz_grid,
                un_theta=neutral_fluid.un_theta_grid,
                ne_floor=electron_fluid.ne_floor,
                Ji_r=Ji_r, Ji_z=Ji_z,
                cathode_voltage_profile=self.phi_cathode_vec,
                phi_anode_value=phi_anode_value,
                phi0=phi0,
                omega=1.8, tol=tol, max_iter=max_iter,
                verbose=verbose
            )

        else:
            phi, info = solve_anisotropic_poisson_FV(
                geom,
                self.sigma_P_grid,
                self.sigma_parallel_grid,
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

        Er, Ez, Jr, Jz, Er_gradpe, Ez_gradpe = compute_E_and_J(phi, geom,
                            self.sigma_P_grid,
                            self.sigma_parallel_grid,
                            ne=electron_fluid.ne_grid,
                            pe=electron_fluid.pe_grid,
                            Bz=self.Bz_grid,
                            un_theta=neutral_fluid.un_theta_grid,
                            ne_floor=electron_fluid.ne_floor,
                            fill_solid_with_nan=False)

        # q_ohm = sigma_P*Er^2 + sigma_parallel*Ez^2
        q_ohm = self.sigma_P_grid*Er*Er + self.sigma_parallel_grid*Ez*Ez

        self.phi_grid = np.copy(phi)
        self.Er_grid = np.copy(Er)
        self.Ez_grid = np.copy(Ez)
        self.Er_grid_grad_pe = np.copy(Er_gradpe)
        self.Ez_grid_grad_pe = np.copy(Ez_gradpe)
        self.Jr_grid = np.copy(Jr)
        self.Jz_grid = np.copy(Jz)
        self.q_ohm_grid = np.copy(q_ohm)

        self.Er_grid_d = cp.asarray(self.Er_grid)
        self.Ez_grid_d = cp.asarray(self.Ez_grid)

        del phi, Er, Ez, Jr, Jz, q_ohm

    def update_Je_Ji_and_compute_DC_power_e_and_i(self, geom, electron_fluid, ion_fluid):
        self.update_Je_and_Ji_from_Jtotal(geom, electron_fluid, ion_fluid)
        # compute q_ohm for electrons and ions separately
        self.q_ohm_electrons_grid = self.Jer_grid*self.Er_grid + self.Jez_grid*self.Ez_grid
        self.q_ohm_ions_grid = self.Jir_grid*self.Er_grid + self.Jiz_grid*self.Ez_grid
        #self.q_ohm_ions_grid = ion_fluid.sigma_P_grid*self.Er_grid*self.Er_grid + ion_fluid.sigma_parallel_grid*self.Ez_grid*self.Ez_grid
        #self.q_ohm_electrons_grid = electron_fluid.sigma_P_grid*self.Er_grid*self.Er_grid + electron_fluid.sigma_parallel_grid*self.Ez_grid*self.Ez_grid

    # -------- Calculate electrodes currents
    def compute_electrode_currents(self, geom, return_parts=False):
        """
        Compute cathode and anode currents (positive = into the electrode) for an axisymmetric RZ grid.

        Parameters
        ----------
        geom : Geometry
            Geometry object (as in your code). Assumes:
            - zmin == 0.0 (domain starts at 0)
            - anode 'upper' radial boundary is at rmax
            - geom.zmax_anode2 already set (zmin_anode2 + (zmax_anode - zmin_anode))
        Jer, Jez : np.ndarray, shape (Nr, Nz)
            Radial and axial current density components [A/m^2].
            Arrays are defined on the same (r,z) nodes as geom.r, geom.z.
        return_parts : bool, default False
            If True, also return a dict with individual face contributions for debugging.

        Returns
        -------
        I_anode : float
            Total current into the anode [A].
        I_cathode : float
            Total current into the cathode [A].
        parts : dict (optional)
            Individual contributions; only returned if return_parts=True.
        """
        # --- helpers ---------------------------------------------------------------
        Jr = self.Jr_grid
        Jz = self.Jz_grid

        twopi = 2.0 * np.pi
        Nr, Nz = geom.Nr, geom.Nz

        def ir_at(rval):
            i = int(round((rval - geom.rmin) / geom.dr))
            return max(0, min(Nr-1, i))

        def iz_at(zval):
            i = int(round((zval - geom.zmin) / geom.dz))
            return max(0, min(Nz-1, i))

        def trapz_r(f_r, r_slice):
            return np.trapz(f_r, geom.r[r_slice])

        def trapz_z(f_z, z_slice):
            return np.trapz(f_z, geom.z[z_slice])

        # indices (use rounding; all your geometry values are multiples of dr/dz)
        assert abs(geom.zmin) < 1e-12, "This function assumes zmin == 0."

        ir0 = 0
        ir_rmax = Nr - 1
        ir_rcath = ir_at(geom.rmax_cathode)
        iz_zcath = iz_at(geom.zmax_cathode)

        ir_rminA = ir_at(geom.rmin_anode)
        iz_zminA1 = iz_at(geom.zmin_anode)
        iz_zmaxA1 = iz_at(geom.zmax_anode)

        iz_zminA2 = iz_at(geom.zmin_anode2)
        # Make sure zmax_anode2 is available (your Geometry sets it)
        if not hasattr(geom, 'zmax_anode2'):
            geom.zmax_anode2 = geom.zmin_anode2 + (geom.zmax_anode - geom.zmin_anode)
        iz_zmaxA2 = iz_at(geom.zmax_anode2)

        iz_zmax = Nz - 1

        parts = {}

        # ------------------- Cathode faces -------------------
        # Top face of the cathode rectangle: z = zmax_cathode, r in [0, rmax_cathode]
        # Sample at the plasma side (above): iz = iz_zcath + 1 (clamped)
        iz_top_plasma = min(iz_zcath + 1, Nz - 1)
        r_slice_cath = slice(0, ir_rcath + 1)
        I_cathode_z = trapz_r(twopi * geom.r[r_slice_cath] * Jz[r_slice_cath, iz_top_plasma], r_slice_cath)

        # Side face of the cathode rectangle: r = rmax_cathode, z in [0, zmax_cathode]
        # Plasma is at larger r, so sample at ir = ir_rcath + 1 (clamped)
        ir_side_plasma = min(ir_rcath + 1, Nr - 1)
        z_slice_cath = slice(0, iz_zcath + 1)
        I_cathode_r = trapz_z(twopi * geom.rmax_cathode * Jr[ir_side_plasma, z_slice_cath], z_slice_cath)

        I_cathode = I_cathode_z + I_cathode_r
        parts['I_cathode_z'] = I_cathode_z
        parts['I_cathode_r'] = I_cathode_r

        # ------------------- Anode ring 1 -------------------
        # Vertical inner face at r = rmin_anode, z in [zmin_anode, zmax_anode]:
        # Plasma is at smaller r, so take ir = ir_rminA (no +1)
        z_slice_A1 = slice(iz_zminA1, iz_zmaxA1 + 1)
        I_a1_r = trapz_z(twopi * geom.rmin_anode * Jr[ir_rminA, z_slice_A1], z_slice_A1)

        # Bottom face at z = zmin_anode, r in [rmin_anode, rmax]; plasma below -> sample at iz = iz_zminA1
        r_slice_A1 = slice(ir_rminA, ir_rmax + 1)
        I_a1_zbot = trapz_r(twopi * geom.r[r_slice_A1] * Jz[r_slice_A1, iz_zminA1], r_slice_A1)

        # Top face at z = zmax_anode, r in [rmin_anode, rmax]; plasma above -> sample at iz = iz_zmaxA1 + 1; sign negative
        iz_A1_top_plasma = min(iz_zmaxA1 + 1, Nz - 1)
        I_a1_ztop = -trapz_r(twopi * geom.r[r_slice_A1] * Jz[r_slice_A1, iz_A1_top_plasma], r_slice_A1)

        parts['I_anode_1_r'] = I_a1_r
        parts['I_anode_1_zbot'] = I_a1_zbot
        parts['I_anode_1_ztop'] = I_a1_ztop

        # ------------------- Anode ring 2 -------------------
        # Vertical inner face at r = rmin_anode, z in [zmin_anode2, zmax_anode2]
        z_slice_A2 = slice(iz_zminA2, iz_zmaxA2 + 1)
        I_a2_r = trapz_z(twopi * geom.rmin_anode * Jr[ir_rminA, z_slice_A2], z_slice_A2)

        # Bottom face at z = zmin_anode2, r in [rmin_anode, rmax]; plasma below
        I_a2_zbot = trapz_r(twopi * geom.r[r_slice_A1] * Jz[r_slice_A1, iz_zminA2], r_slice_A1)

        # Top face at z = zmax_anode2, r in [rmin_anode, rmax]; plasma above -> sample at iz = iz_zmaxA2 + 1; negative sign
        iz_A2_top_plasma = min(iz_zmaxA2 + 1, Nz - 1)
        I_a2_ztop = -trapz_r(twopi * geom.r[r_slice_A1] * Jz[r_slice_A1, iz_A2_top_plasma], r_slice_A1)

        parts['I_anode_2_r'] = I_a2_r
        parts['I_anode_2_zbot'] = I_a2_zbot
        parts['I_anode_2_ztop'] = I_a2_ztop

        # ------------------- Outer wall segments at r = rmax -------------------
        # Segment between rings: z in (zmax_anode, zmin_anode2)
        z_start = min(iz_zmaxA1 + 1, Nz - 1)
        z_stop  = max(iz_zminA2, z_start)  # exclusive in python slice semantics
        I_awall_1 = 0.0
        if z_stop > z_start:
            z_slice_w1 = slice(z_start, z_stop)
            I_awall_1 = trapz_z(twopi * geom.rmax * Jr[ir_rmax, z_slice_w1], z_slice_w1)

        # Segment after ring 2: z in (zmax_anode2, zmax]
        z_start2 = min(iz_zmaxA2 + 1, Nz - 1)
        z_slice_w2 = slice(z_start2, Nz)
        I_awall_2 = 0.0
        if z_start2 < Nz:
            I_awall_2 = trapz_z(twopi * geom.rmax * Jr[ir_rmax, z_slice_w2], z_slice_w2)

        parts['I_anode_wall_between'] = I_awall_1
        parts['I_anode_wall_top'] = I_awall_2

        I_anode = (I_a1_r + I_a1_zbot + I_a1_ztop +
                I_a2_r + I_a2_zbot + I_a2_ztop +
                I_awall_1 + I_awall_2)

        if return_parts:
            return I_anode, I_cathode, parts
        return I_anode, I_cathode

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