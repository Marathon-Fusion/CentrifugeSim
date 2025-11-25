import numpy as np

from centrifugesim.fluids.neutral_fluid import NeutralFluidContainer
from centrifugesim.fluids import electron_fluid_helper
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class ElectronFluidContainer:
    """
    Notes:
        When subcycling electrons, use proper dt for diffusion + Joule heating vs collisions
        Need to update Te advance due to div(kappa*grad(Te)) term using ADI-Douglas or another
        implicit / semi implicit algorithm. Using simple Euler here to test but should not use
        this version for production runs.

        Move diffusion and advection terms to dolfinx!
        Do advance 
    """
    def __init__(self, geom:Geometry, ne_floor, Te_floor):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        # floor values
        self.ne_floor = ne_floor
        self.Te_floor = Te_floor

        self.ne_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !
        self.Te_grid_prev = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !

        self.pe_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa] !
        self.grad_pe_grid_r = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa/m] !
        self.grad_pe_grid_z = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa/m] !

        self.nu_ei_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_en_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_e_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.sigma_P_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.sigma_parallel_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.sigma_H_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.kappa_parallel_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.kappa_perp_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.beta_e_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        # electron drift velocity components (for particle pusher)
        self.uer_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.uet_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.uez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

    def update_drift_velocities(self, hybrid_pic):
        """
        Update electron drift velocities from current densities and ne:
            u_e = J_e / ( -e * n_e )
        """
        qe = constants.q_e  # electron charge (C)

        ne_eff = np.maximum(self.ne_grid, self.ne_floor)  # m^-3 (avoid div-by-zero)

        self.uer_grid[:, :] = hybrid_pic.Jer_grid / (-qe * ne_eff)
        #self.uet_grid[:, :] = hybrid_pic.Jet_grid / (-qe * ne_eff)
        self.uez_grid[:, :] = hybrid_pic.Jez_grid / (-qe * ne_eff)

    def update_pressure(self):
        self.pe_grid = constants.kb*self.Te_grid*self.ne_grid

    def set_kappa(self, hybrid_pic):
        """
        Update electron thermal conductivities (W / m / K),
        including e-i and e-n collisions:
            kappa_parallel = C * n_e * kB^2 * T_e * tau_e / m_e
            kappa_perp     = kappa_parallel / (1 + (Omega_e * tau_e)^2)

        where tau_e = 1 / (nu_ei + nu_en), Omega_e = e * |B| / m_e.
        """
        # Short-hands
        ne = self.ne_grid
        Te = self.Te_grid

        # Physical constants
        kb = constants.kb
        me = constants.m_e
        qe = constants.q_e

        # Numerically safe floors
        ne_eff = np.maximum(ne, self.ne_floor)        # m^-3 (very small floor just to avoid div-by-zero)
        Te_eff = np.maximum(Te, self.Te_floor)       # K
        nu_e   = np.maximum(self.nu_e_grid, 1e-30)  # total electron momentum-transfer frequency
        tau_e  = 1.0 / nu_e

        # Electron gyrofrequency
        Omega_e = np.abs(qe) * hybrid_pic.Bmag_grid / me

        # Parallel electron thermal conductivity (Spitzer–Härm with collisions folded into tau_e)
        Ck = 3.16  # Braginskii/Spitzer-Härm coefficient for electrons
        kappa_par = Ck * ne_eff * (kb*kb) * Te_eff * tau_e / me  # W / m / K

        # Perpendicular electron thermal conductivity (classical suppression by magnetization)
        chi2 = (Omega_e * tau_e)**2
        kappa_perp = kappa_par / (1.0 + chi2)

        # Write back
        self.kappa_parallel_grid[:, :] = kappa_par
        self.kappa_perp_grid[:, :]     = kappa_perp

    def set_electron_collision_frequencies(
        self, nn_grid, lnLambda=10.0, sigma_en_m2=5e-20, Te_is_eV=False
    ):
        """
        Compute and set electron collision frequencies:
          - self.nu_en : electron-neutral momentum-transfer
          - self.nu_ei : electron-ion (Spitzer)
          - self.nu_e  : total = nu_en + nu_ei
        """
        nu_en_grid, nu_ei_grid, nu_e_grid = electron_fluid_helper.electron_collision_frequencies(
            self.Te_grid, self.ne_grid, nn_grid, lnLambda=lnLambda, sigma_en_m2=sigma_en_m2, Te_is_eV=Te_is_eV
        )
        self.nu_en_grid[:] = nu_en_grid
        self.nu_ei_grid[:] = nu_ei_grid
        self.nu_e_grid[:]  = nu_e_grid

    def set_electron_conductivities(
        self, hybrid_pic, lnLambda=10.0, sigma_en_m2=5e-20, Te_is_eV=False
    ):
        """
        Compute and set electron conductivity tensor components:
          - self.sigma_parallel, self.sigma_P, self.sigma_H
        """

        Bmag_grid = hybrid_pic.Bmag_grid

        sigma_par_e, sigma_P_e, sigma_H_e, _beta_e = electron_fluid_helper.electron_conductivities(
            self.Te_grid, self.ne_grid, Bmag_grid, self.nu_e_grid, lnLambda=lnLambda, sigma_en_m2=sigma_en_m2, Te_is_eV=Te_is_eV
        )
        self.sigma_parallel_grid[:] = sigma_par_e
        self.sigma_P_grid[:]        = sigma_P_e
        self.sigma_H_grid[:]        = sigma_H_e
        self.beta_e_grid[:]         = _beta_e

    def update_Te(self, geom, hybrid_pic, neutral_fluid, Q_Joule_grid, dt):
        "Update Te function solving energy equation"
        Te_new = np.zeros_like(self.Te_grid)

        # Effective thermal diffusivity for electrons from parallel conductivity
        # D_eff = (2/3) * kappa_parallel / (n_e * k_B)
        with np.errstate(divide='ignore', invalid='ignore'):
            D_eff = (2.0 * self.kappa_parallel_grid) / (3.0 * self.ne_grid * constants.kb)

        # Zero diffusion where n_e is below floor
        D_eff = np.where(self.ne_grid < self.ne_floor, 0.0, D_eff)

        # Stable explicit timestep for 2D diffusion-like operator:
        # dt_stable = 1 / ( 2 * D_max * (1/dr^2 + 1/dz^2) )
        inv_h2 = (1.0 / geom.dr**2) + (1.0 / geom.dz**2)
        Dmax = float(np.nanmax(D_eff)) if np.isfinite(D_eff).any() else 0.0

        if Dmax > 0.0 and np.isfinite(Dmax) and inv_h2 > 0.0:
            dt_stable = 1.0 / (2.0 * Dmax * inv_h2)
        else:
            dt_stable = dt  # no diffusion -> no stability restriction

        Q_Joule_grid = np.where(self.ne_grid<self.ne_floor, 0, Q_Joule_grid)

        # Helper to perform one advance with a given local dt
        def _advance(dt_local):
            electron_fluid_helper.solve_step(
                self.Te_grid, Te_new,
                geom.dr, geom.dz, geom.r,
                self.ne_grid, Q_Joule_grid,
                hybrid_pic.br_grid, hybrid_pic.bz_grid,
                self.kappa_parallel_grid, self.kappa_perp_grid,
                hybrid_pic.Jer_grid, hybrid_pic.Jez_grid,
                geom.mask, dt_local
            )
            self.Te_grid[:, :] = Te_new

            # Enforce Dirichlet BCs at cathode and anode
            # This is just to test, should change to sheath based model!
            self.Te_grid[geom.cathode_mask] = geom.temperature_cathode
            self.Te_grid[geom.anode1_mask] = geom.temperature_anode
            self.Te_grid[geom.anode2_mask] = geom.temperature_anode

            # BCs at rmin, rmax, zmin, zmax
            # Using Neumann here, should change to sheath based model!
            self.apply_boundary_conditions()

        # Sub-stepping controller
        if not np.isfinite(dt_stable) or dt_stable <= 0.0:
            dt_stable = dt

        if dt_stable < dt:
            # Full sub-steps of size dt_stable
            n_full = int(dt // dt_stable)
            t_accum = 0.0
            for _ in range(n_full):
                _advance(dt_stable)
                t_accum += dt_stable
            # Final remainder (if dt is not an exact multiple)
            dt_rem = dt - t_accum
            if dt_rem > 0.0:
                _advance(dt_rem)
        else:
            # Single step with dt
            _advance(dt)

        # self.Te_grid_prev = Te_grid.copy() # saving for ion T relaxation

        # --- Collision energy exchange term uses the full dt (outside of sub-steps) ---
        self.compute_elastic_collisions_term(geom, neutral_fluid, dt)

        self.Te_grid[geom.cathode_mask] = geom.temperature_cathode
        self.Te_grid[geom.anode1_mask] = geom.temperature_anode
        self.Te_grid[geom.anode2_mask] = geom.temperature_anode

        self.apply_boundary_conditions()

    def compute_elastic_collisions_term(
        self,
        geom,
        neutral_fluid,
        dt,
        cap=0.1
    ):
        """
        Collisional Energy Exchange
        - Keeping only neutral gas here.
        - for collisions with ions should update ion Ti too but using particle ions
          so might have to move to the drag diffusion part to keep it consistent..
          It might have to use a much smaller timestep
        """
        ne = np.where(self.ne_grid<self.ne_floor,self.ne_floor,self.ne_grid)
        nn = np.where(neutral_fluid.nn_grid<neutral_fluid.nn_floor, neutral_fluid.nn_floor, neutral_fluid.nn_grid)

        m_ratio_n = constants.m_e/neutral_fluid.mass
        
        # Substep count so that both nu_ei*dt_sub and nu_en*dt_sub <= cap (as requested)
        max_nu_dt = 0.0
        max_nu_dt = max(max_nu_dt, float(np.nanmax(self.nu_en_grid * m_ratio_n * dt)))
        n_sub = int(np.ceil(max(1.0, max_nu_dt / cap)))
        dt_sub = dt / n_sub

        # Subcycling with operator splitting: e–i then e–n each substep
        for _ in range(n_sub):
            Q_coll_en = 3 * self.ne_grid * constants.kb * (
                m_ratio_n * self.nu_en_grid * (self.Te_grid - neutral_fluid.T_n_grid) )

            de = dt_sub*Q_coll_en[geom.mask==1] # J/m^3

            # Write back masked regions
            self.Te_grid[geom.mask==1] -= de/(3/2*constants.kb*ne[geom.mask==1])
            neutral_fluid.T_n_grid[geom.mask==1] += de/(3/2*constants.kb*nn[geom.mask==1])


    def apply_boundary_conditions(self):
        """
        Update this!
        Add proper BCs for cathode and anode based on sheath physics
        Use geometry object and update with 
        """
        # Axis of symmetry (r=0): dTe/dr = 0
        self.Te_grid[0, :] = self.Te_grid[1, :]

        # Outer walls: zero-flux
        self.Te_grid[-1, :] = self.Te_grid[-2, :]
        self.Te_grid[:, 0] = self.Te_grid[:, 1]
        self.Te_grid[:, -1] = self.Te_grid[:, -2]