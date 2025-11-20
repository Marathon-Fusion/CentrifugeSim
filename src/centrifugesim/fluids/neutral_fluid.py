import numpy as np

from centrifugesim.fluids import neutral_fluid_helper
from centrifugesim.geometry.geometry import Geometry

from centrifugesim import constants

class NeutralFluidContainer:
    """
    """
    def __init__(self, geom:Geometry, species_list, nn_floor, mass, name, kind, Tn0=0.0):

        self.name = name

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.nn_floor = nn_floor
        self.mass = mass

        self.kind = kind
        if(self.kind=='monatomic'):
            gamma = 5/3.0
        elif(self.kind=='diatomic'):
            gamma = 7/5.0

        self.gamma = gamma
        self.Rgas_over_m = constants.kb/self.mass # J/(kg·K) 
        self.c_v = self.Rgas_over_m/(gamma - 1.0) # J/(kg·K)
        self.cp  = self.c_v + self.Rgas_over_m # J/(kg·K)

        # For ground/excited states
        self.str_states_list = species_list.copy()
        self.list_nn_grid = [np.zeros((self.Nr, self.Nz)).astype(np.float64) for _ in species_list]

        self.nn_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.rho_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.p_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.un_r_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_theta_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_z_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        
        self.T_n_grid = (Tn0*np.ones((self.Nr, self.Nz))).astype(np.float64)
        self.mu_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.kappa_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.fluid = (geom.mask.astype(np.uint8)).copy()
        self.face_r, self.face_z = neutral_fluid_helper.build_face_masks(self.fluid)

        self.i_bc_list, self.j_bc_list = geom.i_bc_list.copy(), geom.j_bc_list.copy()

        print("initialized")
        print(self.str_states_list)
        print()

    def initialize_state(self, state_name, value):
        idx = self.str_states_list.index(state_name)
        self.list_nn_grid[idx][:,:] = value

    def compute_nn_grid_from_states(self):
        self.nn_grid[:,:] = 0.0
        for i, species in enumerate(self.str_states_list):
            self.nn_grid[:,:] += self.list_nn_grid[i][:,:]

    def update_rho(self):
        self.rho_grid[self.fluid==1] = self.mass*self.nn_grid[self.fluid==1]

    def update_nn(self):
        self.nn_grid[self.fluid==1] = self.rho[self.fluid==1]/self.mass

    def update_p(self):
        self.p_grid[self.fluid==1] = self.rho_grid[self.fluid==1] * self.Rgas_over_m * self.T_n_grid[self.fluid==1]

    def compute_sound_speed(self, Tfield):
        return np.sqrt(self.gamma * self.Rgas_over_m * Tfield)

    def update_u_in_collisions(self, geom, ni_grid, mi,
                                    ui_r, ui_t, ui_z,
                                    nu_in, Ti, dt):

        dtnu_max = nu_in[geom.mask==1].max()*dt
        if(round(dtnu_max,3)>0.1):
            print("dt*nu_in.max() > 0.1 !", dtnu_max)

        un_r_new, un_t_new, un_z_new, Tn_new = neutral_fluid_helper.update_u_in_collisions(
                                        geom.mask, ni_grid*mi, self.rho_grid,
                                        ui_r, ui_t, ui_z,
                                        self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                        nu_in, self.nn_floor*self.mass,
                                        self.T_n_grid, Ti, self.c_v, dt)
                                        
        self.un_r_grid[geom.mask==1]        = np.copy(un_r_new[geom.mask==1])
        self.un_theta_grid[geom.mask==1]    = np.copy(un_t_new[geom.mask==1])
        self.un_z_grid[geom.mask==1]        = np.copy(un_z_new[geom.mask==1])
        self.T_n_grid[geom.mask==1]         = np.copy(Tn_new[geom.mask==1])

    ############################### NEW CLEANUP #################################

    def advance_with_T_ssp_rk2(self,
                            geom, dt,
                            c_iso,
                            apply_bc_vel, apply_bc_temp,
                            T_wall=0):

        r, dr, dz = geom.r, geom.dr, geom.dz

        # stage-0
        rho0 = self.rho_grid.copy()
        ur0 = self.un_r_grid.copy()
        ut0 = self.un_theta_grid.copy()
        uz0 = self.un_z_grid.copy()
        T0 = self.T_n_grid.copy()

        mub = np.zeros_like(T0)

        # ---------- stage 1: momentum + continuity
        neutral_fluid_helper.step_isothermal(r, dr, dz, dt,
                    self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                    self.p_grid, self.mu_grid, mub,
                    c_iso,
                    fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        # stresses for energy (use your masked version if you added one)
        tau_rr = np.zeros_like(rho0); tau_tt = np.zeros_like(rho0); tau_zz = np.zeros_like(rho0)
        tau_rz = np.zeros_like(rho0); tau_rt = np.zeros_like(rho0); tau_tz = np.zeros_like(rho0)
        divu_d = np.zeros_like(rho0)
        neutral_fluid_helper.stresses(r, self.un_r_grid, self.un_theta_grid, self.un_z_grid, self.mu_grid, mub, dr, dz,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu_d,
            fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        neutral_fluid_helper.step_temperature_masked(r, dr, dz, dt,
                                self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                self.p_grid, self.kappa_grid,
                                tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                                self.c_v,
                                fluid=self.fluid, face_r=self.face_r, face_z=self.face_z,
                                T_floor=1.0, T_wall=T_wall)

        # EOS + BCs + projection
        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()
        if self.fluid is not None:
            neutral_fluid_helper.apply_solid_mask_inplace_T(self.fluid, self.T_n_grid, T_wall)

        # ---------- stage 2: repeat
        neutral_fluid_helper.step_isothermal(r, dr, dz, dt,
                    self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                    self.p_grid, self.mu_grid, mub,
                    c_iso,
                    fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        tau_rr.fill(0.0); tau_tt.fill(0.0); tau_zz.fill(0.0)
        tau_rz.fill(0.0); tau_rt.fill(0.0); tau_tz.fill(0.0); divu_d.fill(0.0)
        neutral_fluid_helper.stresses(r, self.un_r_grid, self.un_theta_grid, self.un_z_grid, 
            self.mu_grid, mub, dr, dz,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu_d,
            fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        neutral_fluid_helper.step_temperature_masked(r, dr, dz, dt,
                                self.T_n_grid, self.rho_grid,
                                self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                self.p_grid, self.kappa_grid,
                                tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                                self.c_v,
                                fluid=self.fluid, face_r=self.face_r, face_z=self.face_z,
                                T_floor=1.0, T_wall=T_wall)

        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()
        if self.fluid is not None:
            neutral_fluid_helper.apply_solid_mask_inplace_T(self.fluid, self.T_n_grid, T_wall)

        # ---------- combine
        self.rho_grid[:,:]          = 0.5*(rho0 + self.rho_grid)
        self.un_r_grid [:,:]        = 0.5*(ur0  + self.un_r_grid)
        self.un_theta_grid [:,:]    = 0.5*(ut0  + self.un_theta_grid)
        self.un_z_grid [:,:]        = 0.5*(uz0  + self.un_z_grid)
        self.T_n_grid[:,:]          = 0.5*(T0   + self.T_n_grid)
        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()
        if self.fluid is not None:
            neutral_fluid_helper.apply_solid_mask_inplace_T(self.fluid, self.T_n_grid, T_wall)

    # --------------------------- Boundary conditions -------------------------

    def apply_bc_isothermal(self):
        Nr, Nz = self.rho_grid.shape

        # --- Axis r = 0 (i = 0): regularity ---
        self.un_r_grid[0,:]  = 0.0                 # odd
        self.un_theta_grid[0,:]  = 0.0                 # odd
        self.un_z_grid[0,:]  = self.un_z_grid[1,:]             # ∂r uz = 0
        self.rho_grid[0,:] = self.rho_grid[1,:]            # ∂r rho = 0
        self.p_grid[0,:]   = self.p_grid[1,:]              # ∂r p   = 0

        # --- Radial wall r = R (i = Nr-1): no-slip, impermeable ---
        self.un_r_grid[-1,:] = 0.0
        self.un_theta_grid[-1,:] = 0.0
        self.un_z_grid[-1,:] = 0
        self.rho_grid[-1,:] = self.rho_grid[-2,:]
        self.p_grid[-1,:] = self.p_grid[-2,:]

        # --- Bottom plate z = 0 (k = 0): no-slip, impermeable ---
        self.un_r_grid[:,0] = 0
        self.un_theta_grid[:,0] = 0
        self.un_z_grid[:,0] = 0.0
        self.rho_grid[:,0] = self.rho_grid[:,1]
        self.p_grid[:,0] = self.p_grid[:,1]

        # --- Top plate z = L (k = Nz-1): no-slip, impermeable ---
        self.un_r_grid[:,-1] = self.un_r_grid[:,-2]
        self.un_theta_grid[:,-1] = self.un_theta_grid[:,-2]
        #self.un_r_grid[:,-1] = 0
        #self.un_theta_grid[:,-1] = 0
        
        self.un_z_grid[:,-1] = 0.0
        self.rho_grid[:,-1] = self.rho_grid[:,-2]
        self.p_grid[:,-1] = self.p_grid[:,-2]

        # no slip solid surfaces inside domain
        self.un_r_grid[self.i_bc_list, self.j_bc_list] = 0
        self.un_theta_grid[self.i_bc_list, self.j_bc_list] = 0
        self.un_z_grid[self.i_bc_list, self.j_bc_list] = 0


    # Remove this and add to BC functions above for each case.
    def apply_bc_T(self):
        Nr, Nz = self.T_n_grid.shape
        # r = 0 axis: Neumann
        self.T_n_grid[0,:] = self.T_n_grid[1,:]
        # r = R wall: Neumann
        self.T_n_grid[-1,:]  = self.T_n_grid[-2,:]
        # z 
        self.T_n_grid[:,0]  = self.T_n_grid[:,1]
        self.T_n_grid[:,-1] = self.T_n_grid[:,-2]
    