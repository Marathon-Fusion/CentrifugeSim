import numpy as np
from numba import njit

from centrifugesim.geometry.geometry import Geometry

"""
TO DO :
- Instead of defining coil geometry with a lower point (r,z) and a length dzc and thickness drc
    it should be defined by setting 4 points to allow for different geom in RZ plane
    then for each coil, compute_Jt should be able to fill the i,j nodes inside it. 

- Add functions to update current density and calculate induced voltage from change in Psi.

- Change to have an object per coil and then make a list of coils?
"""

import numpy as np

class Coils:
    """
    Pre-computes masks for a set of rectangular, toroidal coils
    centred on the z-axis in an R-Z grid, and evaluates poloidal
    flux  φ = 2π⟨Ψ⟩  for each coil at runtime.

    Parameters
    ----------
    geom: Geometry object

    coils_dict : dictionary
        Must contain arrays with keys 'rc', 'drc', 'zc', 'dzc' and 'coil' for coil name    
    """

    def __init__(self, geom:Geometry, coils_dict):

        self.coils = coils_dict['coil']
        self.Nc = self.coils.shape[0]

        self.r1c = coils_dict['r1c']
        self.r2c = coils_dict['r2c']
        self.rc = coils_dict['rc']
        self.zc = coils_dict['zc']
        self.drc = coils_dict['drc']
        self.dzc = coils_dict['dzc']

        self.dAc = self.drc*self.dzc # change if coils are not 'rectangular'

        self.Ic = np.zeros(self.Nc).astype(np.float64)
        self.Jtc = np.zeros(self.Nc).astype(np.float64)

        R, Z = geom.R, geom.Z

        # pre‑compute cell areas (RZ, cell‑centre convention)
        dr = geom.dr  # assume uniform; adjust if needed
        dz = geom.dz
        self.cell_area = dr * dz * np.ones((geom.Nr, geom.Nz))

        # build masks – one Boolean array per coil
        # masks is a dictionary of masks to quickly access each coil nodes i,j from a key name (from 'coil' in coils_dict)
        self.masks = {}

        for ic in range(self.Nc):
            # Axial limits ----------------------------------------------------------
            z0 = self.zc[ic]              # lower axial face
            dz = self.dzc[ic]             # axial thickness
            z1 = z0 + dz                  # upper axial face
            inside_z = (Z >= z0) & (Z <= z1)

            # Inner and outer radial faces (linear in Z) ----------------------------
            τ        = (Z - z0) / dz                  # 0 ≤ τ ≤ 1 within the coil
            r_inner  = self.r1c[ic] + (self.r2c[ic] - self.r1c[ic]) * τ
            r_outer  = r_inner + self.drc[ic]         # constant thickness Δr_c
            inside_r = (R >= r_inner) & (R <= r_outer) # check this

            # Combined stair-case mask ---------------------------------------------
            self.masks[self.coils[ic]] = inside_z & inside_r


        # Induced voltage per coil dict
        self.v_induced = {}
        for ic in range(self.Nc):
            self.v_induced[self.coils[ic]] = 0.0

    def set_Ic_from_external_dict(self, ext_dict_currents):
        """
        Set self.Ic from an external dictionary of currents keyed by coil names.

        Parameters
        ----------
        ext_dict_currents : dict
            Dictionary mapping coil names (e.g., 'F09', 'A04') to current values.

        This function safely handles arbitrary ordering of the external keys.
        """
        for ic, name in enumerate(self.coils):
            if name in ext_dict_currents:
                self.Ic[ic] = ext_dict_currents[name]
            else:
                raise KeyError(f"Coil name '{name}' not found in external current dictionary.")

    def compute_Jtc(self):
        """
        Compute coil current density using precomputed coil area and assuming current is uniform.
        """
        self.Jtc = self.Ic/self.dAc