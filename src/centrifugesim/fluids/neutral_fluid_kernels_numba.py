import numpy as np
import numba
from numba import njit

from centrifugesim import constants

@njit(cache=True)
def test_update_u_theta(
    mask, rho_i, rho_n, ui_theta, un_theta, nu, rho_floor, dt
):
    """
    Just to test before merging with compressible Navier Stokes equations.
    This also assumes that dt*nu<0.1
    """
    NR, NZ = mask.shape

    un_theta_new = np.copy(un_theta)
    # Loop over the interior of the grid (parallelized by Numba)
    for i in numba.prange(1, NR - 1):
        for j in numba.prange(1, NZ - 1):
            if(mask[i, j] == 1 and rho_n[i, j]>rho_floor):
                un_theta_new[i, j] = un_theta[i, j] + dt*rho_i[i, j]/rho_n[i, j] * nu[i, j] * (ui_theta[i, j] - un_theta[i, j])

    return un_theta_new