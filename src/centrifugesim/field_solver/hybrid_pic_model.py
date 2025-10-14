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

        self.Br = np.zeros((self.Nr, self.Nz))
        self.Bt = np.zeros((self.Nr, self.Nz))  # unused in solver; kept for pusher
        self.Bz = np.zeros((self.Nr, self.Nz))

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

        # Electron thermal conductivities (placeholders)
        self.kappa_perp = np.zeros((self.Nr, self.Nz))
        self.kappa_parallel = np.zeros((self.Nr, self.Nz))


    # --- Add function to solve for phi ---
    def solve_phi(
        self,
        pe, un_theta, un_r, ne,
        sigma_P=None, sigma_H=None, sigma_parallel=None, Bz=None,
        mask=None, phi_initial=None,
        e_charge=constants.q_e, max_iter=25000, tol=1e-6, omega=1.5
    ):
        """
        Solve the 2D elliptic PDE for the electric potential phi on this geometry.
        Updates self.phi, self.Er, self.Ez, and electron currents self.Jer, self.Jez.
        """
        Nr, Nz = self.Nr, self.Nz
        shape = (Nr, Nz)

        def _ensure(a, fallback):
            return fallback if a is None else a

        # use provided fields or class-held ones
        sigma_P = _ensure(sigma_P, self.sigma_P)
        sigma_H = _ensure(sigma_H, self.sigma_H)
        sigma_parallel = _ensure(sigma_parallel, self.sigma_parallel)
        Bz = _ensure(Bz, self.Bz)

        # basic shape checks
        for name, arr in [
            ("pe", pe), ("un_theta", un_theta), ("un_r", un_r), ("ne", ne),
            ("sigma_P", sigma_P), ("sigma_H", sigma_H), ("sigma_parallel", sigma_parallel),
            ("Bz", Bz)
        ]:
            if not isinstance(arr, np.ndarray) or arr.shape != shape:
                raise ValueError(f"{name} must be a numpy array with shape {shape}")

        if mask is None:
            mask = np.ones(shape, dtype=np.int8)
        elif mask.shape != shape:
            raise ValueError(f"mask must have shape {shape}")

        if phi_initial is None:
            phi_initial = self.phi
        elif phi_initial.shape != shape:
            raise ValueError(f"phi_initial must have shape {shape}")

        # call the compiled solver core
        phi_solution, iters = electron_fluid_kernels_numba._solve_phi_core(
            phi_initial.copy(),
            sigma_P, sigma_H, sigma_parallel,
            pe, Bz, un_theta, un_r, ne,
            self.r, self.dr, self.dz,
            e_charge, max_iter, tol, omega, mask
        )

        # store results
        self.phi[:] = phi_solution

        # update E-field from phi
        Er, Ez = electron_fluid_kernels_numba.compute_electric_field(
            self.phi, self.dr, self.dz, self.r, mask
        )
        self.Er[:] = Er
        self.Ez[:] = Ez
        # self.Et remains zero by design

        # update J-field (electron current density) using kernels
        Jr, Jz, _, _ = electron_fluid_kernels_numba.compute_currents(
            self.phi, sigma_P, sigma_H, sigma_parallel,
            pe, Bz, un_theta, un_r, ne,
            self.dr, self.dz, self.r, e_charge, mask
        )
        self.Jer[:] = Jr
        self.Jez[:] = Jz

        return phi_solution, iters
    

    def compute_q_ohm(
        self,
        pe, un_theta, un_r, ne,
        sigma_P=None, sigma_H=None, sigma_parallel=None, Bz=None,
        mask=None, e_charge=constants.q_e
    ):
        """
        Return the non-negative Joule heating per unit volume (q_ohm) on this grid.

        q_ohm = sigma_P*(E_r + S_r)^2 + sigma_parallel*(E_z + S_z)^2,
        where E = -∇phi and S are the source terms used by the kernels.
        """
        Nr, Nz = self.Nr, self.Nz
        shape = (Nr, Nz)

        def _ensure(a, fallback):
            return fallback if a is None else a

        sigma_P = _ensure(sigma_P, self.sigma_P)
        sigma_H = _ensure(sigma_H, self.sigma_H)
        sigma_parallel = _ensure(sigma_parallel, self.sigma_parallel)
        Bz = _ensure(Bz, self.Bz)

        # Basic shape checks
        for name, arr in [
            ("pe", pe), ("un_theta", un_theta), ("un_r", un_r), ("ne", ne),
            ("sigma_P", sigma_P), ("sigma_H", sigma_H), ("sigma_parallel", sigma_parallel),
            ("Bz", Bz)
        ]:
            if not isinstance(arr, np.ndarray) or arr.shape != shape:
                raise ValueError(f"{name} must be a numpy array with shape {shape}")

        if mask is None:
            mask = np.ones(shape, dtype=np.int8)
        elif mask.shape != shape:
            raise ValueError(f"mask must have shape {shape}")

        q_ohm, _q_raw = electron_fluid_kernels_numba.joule_heating(
            self.phi, sigma_P, sigma_H, sigma_parallel,
            pe, Bz, un_theta, un_r, ne,
            self.dr, self.dz, self.r, e_charge, mask
        )
        return q_ohm


    def set_electron_collision_frequencies(
        self, Te, ne, nn, lnLambda=10.0, sigma_en_m2=5e-20, Te_is_eV=False
    ):
        """
        Compute and set electron collision frequencies:
          - self.nu_en : electron-neutral momentum-transfer
          - self.nu_ei : electron-ion (Spitzer)
          - self.nu_e  : total = nu_en + nu_ei
        """
        nu_en, nu_ei, nu_e = electron_fluid_kernels_numba.electron_collision_frequencies(
            Te, ne, nn, lnLambda=lnLambda, sigma_en_m2=sigma_en_m2, Te_is_eV=Te_is_eV
        )
        self.nu_en[:] = nu_en
        self.nu_ei[:] = nu_ei
        self.nu_e[:]  = nu_e
        return self.nu_en, self.nu_ei, self.nu_e


    def set_electron_conductivities(
        self, Te, ne, nn, Br=None, Bz=None, lnLambda=10.0, sigma_en_m2=5e-20, Te_is_eV=False
    ):
        """
        Compute and set electron conductivity tensor components:
          - self.sigma_parallel, self.sigma_P, self.sigma_H
        """
        Br = self.Br if Br is None else Br
        Bz = self.Bz if Bz is None else Bz

        sigma_par_e, sigma_P_e, sigma_H_e, _beta_e = electron_fluid_kernels_numba.electron_conductivities(
            Te, ne, nn, Br, Bz, lnLambda=lnLambda, sigma_en_m2=sigma_en_m2, Te_is_eV=Te_is_eV
        )
        self.sigma_parallel[:] = sigma_par_e
        self.sigma_P[:]        = sigma_P_e
        self.sigma_H[:]        = sigma_H_e
        return self.sigma_P, self.sigma_parallel, self.sigma_H