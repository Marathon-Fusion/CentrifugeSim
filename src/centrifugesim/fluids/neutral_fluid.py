import numpy as np

from centrifugesim.fluids import neutral_fluid_kernels_numba
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class NeutralFluidContainer:
    """
    Notes:
    
    """
    def __init__(self, geom:Geometry, flow_rate, r_max_flow, z_max_flow):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.flow_rate = flow_rate
        self.r_max_flow = r_max_flow
        self.z_max_flow = z_max_flow