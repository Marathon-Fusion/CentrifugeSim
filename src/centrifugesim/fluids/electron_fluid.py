import numpy as np

from centrifugesim.fluids import electron_fluid_kernels_numba
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class ElectronFluidContainer:
    """
    Notes:
        when subcycling electrons, use proper dt for diffusion + Joule heating vs collisions
        Need to update Te advance due to div(kappa*grad(Te)) term using ADI-Douglas or another
        implicit / semi implicit algorithm. Using simple Euler here to test but should not use
        this version for production runs.
    """
    def __init__(self, geom:Geometry):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.ne = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !
        self.pe = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa] !

        self.kappa_parallel = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.kappa_perp = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

    def update_pressure(self):
        self.pe = constants.kb*self.Te*self.ne

    def update_kappa(self, hybrid_pic):
        """
        Update electron thermal conductivities (W / m / K),
        including e-i and e-n collisions:
            kappa_parallel = C * n_e * kB^2 * T_e * tau_e / m_e
            kappa_perp     = kappa_parallel / (1 + (Omega_e * tau_e)^2)

        where tau_e = 1 / (nu_ei + nu_en), Omega_e = e * |B| / m_e.
        """
        # Short-hands
        ne = self.ne
        Te = self.Te
        nu_ei = hybrid_pic.nu_ei      # [1/s], shape (Nr, Nz)
        nu_en = hybrid_pic.nu_en      # [1/s], shape (Nr, Nz)
        Bmag = hybrid_pic.Bmag

        # Physical constants
        kb = constants.kb
        me = constants.m_e
        qe = constants.e

        # Numerically safe floors
        ne_eff = np.maximum(ne, hybrid_pic.ne_floor)        # m^-3 (very small floor just to avoid div-by-zero)
        Te_eff = np.maximum(Te, hybrid_pic.Te_floor)       # K
        nu_e   = np.maximum(nu_ei + nu_en, 1e-30)  # total electron momentum-transfer frequency
        tau_e  = 1.0 / nu_e

        # Electron gyrofrequency
        Omega_e = np.abs(qe) * Bmag / me

        # Parallel electron thermal conductivity (Spitzer–Härm with collisions folded into tau_e)
        Ck = 3.16  # Braginskii/Spitzer-Härm coefficient for electrons
        kappa_par = Ck * ne_eff * (kb*kb) * Te_eff * tau_e / me  # W / m / K

        # Perpendicular electron thermal conductivity (classical suppression by magnetization)
        chi2 = (Omega_e * tau_e)**2
        kappa_perp = kappa_par / (1.0 + chi2)

        # Write back
        self.kappa_parallel[:, :] = kappa_par
        self.kappa_perp[:, :]     = kappa_perp

    def update_Te(self, geometry, hybrid_pic, Q_Joule, dt):
        "Update Te function solving energy equation"
        ni = self.ne
        Te_new = np.zeros_like(self.Te)

        Te_new = electron_fluid_kernels_numba.solve_step(self.Te, Te_new,
                                                        geometry.dr, geometry.dz, geometry.r,
                                                        self.ne, Q_Joule,
                                                        hybrid_pic.br, hybrid_pic.bz, self.kappa_parallel, self.kappa_perp,
                                                        geometry.mask, dt)
        self.Te[:,:] = np.copy(Te_new)
    
    def compute_elastic_collisions_term(self, T_i, T_n, nu_ei, nu_en, m_i, m_n, dt):
        """
        Collisional Energy Exchange
        """
        m_ratio_i = constants.m_e/m_i
        m_ratio_n = constants.m_e/m_n
        Q_coll = 3 * self.ne * constants.kb * (
            m_ratio_i * nu_ei * (self.Te - T_i) +
            m_ratio_n * nu_en * (self.Te - T_n)
        )
        self.Te -= dt*Q_coll/(3/2*constants.kb*self.ne)

    def apply_boundary_conditions(self):
        """
        Update this!
        Add proper BCs for cathode and anode based on sheath physics
        Use geometry object and update with 
        """
        # Axis of symmetry (r=0): dTe/dr = 0
        self.Te[0, :] = self.Te[1, :]

        # Outer walls: zero-flux
        self.Te[-1, :] = self.Te[-2, :]
        self.Te[:, 0] = self.Te[:, 1]
        self.Te[:, -1] = self.Te[:, -2]

        # Fix above, set heat flux to electrodes and dielectric, no flux only on z+ and axis.