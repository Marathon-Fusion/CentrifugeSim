import numpy as np

from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

# Write electron fluid class to include:
    # electron energy equation for Te
    # electron density (from quasineutrality)
    # Ohms law goes into hybrid_pic


class ElectronFluidContainer:
    def __init__(self, geom:Geometry):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.ne = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !

        self.kappa_parallel = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.kappa_perp = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

    def initialize_Te(self):
        None

    def update_Te(self):
        "Update Te function, will call add_Joule_heating, thermal transport"
        None

    def add_Joule_heating(self):
        None

    def add_transport(self):
        None