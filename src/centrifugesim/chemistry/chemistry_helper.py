from numba import njit, prange
import numpy as np

from centrifugesim import constants

@njit(parallel=True, cache=True)
def compute_dTe_inelastic(mask, ne, R_per_vol, energy_per_reaction_J, dt, ne_floor):
    """
    Compute the change in electron temperature due to inelastic process
    """
    dTe = np.zeros_like(ne, dtype=np.float64)
    Nr, Nz = ne.shape
    coef = 1.5 * constants.kb

    if(energy_per_reaction_J != 0.0):
        for i in prange(Nr):
            for j in range(Nz):
                if mask[i, j] == 1 and ne[i, j] >= ne_floor:
                    denom = coef * ne[i, j]
                    dTe[i, j] = dt * (R_per_vol[i, j] * energy_per_reaction_J) / denom

    return dTe