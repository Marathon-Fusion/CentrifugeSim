from numba import njit, prange
import numpy as np

from centrifugesim import constants

@njit(parallel=True, cache=True)
def update_Te_inelastic(mask, Te, ne, nn, k_ion, delta_E_J, dt, ne_floor):
    """
    Parallel masked update for electron temperature from ionization energy loss.

    Parameters
    ----------
    mask : 2D int array (0/1)
    Te, ne, nn, k_ion : 2D float arrays (same shape)
    delta_E_J : float
        Ionization energy per reaction [J].
    dt : float
    ne_floor : float
        Minimum electron density to avoid division by zero.

    Returns
    -------
    Te_out : 2D float array
        Updated electron temperature [K].
    ne_nn_k_dt : 2D float array
        k_ion * ne * nn * dt, masked (0 where mask==0).
    """
    n0, n1 = Te.shape
    Te_out = Te.copy()
    ne_nn_k_dt = np.zeros_like(Te)
    denom_factor = 1.5 * constants.kb  # 3/2 * k_b

    for i in prange(n0):
        for j in range(n1):
            if mask[i, j] == 1:
                # Guard against near-vacuum to avoid division by zero / huge steps
                ne_eff = ne[i, j]
                if ne_eff < ne_floor:
                    ne_eff = ne_floor

                # Power sink from ionization: ne * nn * k * ΔE
                P_ion = ne[i, j] * nn[i, j] * k_ion[i, j] * delta_E_J

                # Explicit Euler step on Te:
                dT = dt * (- P_ion) / (denom_factor * ne_eff)
                Te_tmp = Te[i, j] + dT

                # Clamp to non-negative temperatures (numerical safety)
                if Te_tmp < 0.0:
                    Te_tmp = 0.0

                Te_out[i, j] = Te_tmp

                # Save requested field
                if(ne[i, j]>=ne_floor):
                    ne_nn_k_dt[i, j] = k_ion[i, j] * ne[i, j] * nn[i, j] * dt

    return Te_out, ne_nn_k_dt