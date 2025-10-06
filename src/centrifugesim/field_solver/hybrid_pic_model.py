import numpy as np


# This object will have everything needed to solve for the generalized Ohms law
# to get the electric potential and field

# Ohms law solver for electric potential goes here
# Inputs should include
#       resistivity tensor or helper to calculate based on chosen model
#       external magnetic field
#       neutral fluid velocity field
#       geometry


class HybridPICModel:
    def __init__(self, geometry):
        

        # change this to just geometry object as input in functions?
        self.zmin = geometry.zmin
        self.Nr = geometry.Nr
        self.Nz = geometry.Nz
        self.dr = geometry.dr
        self.dz = geometry.dz
        self.r = geometry.r

        self.phi = np.zeros((self.Nr, self.Nz))
        
        self.Er = np.zeros((self.Nr, self.Nz))
        self.Ez = np.zeros((self.Nr, self.Nz))
        
        self.Br = np.zeros((self.Nr, self.Nz))
        self.Bz = np.zeros((self.Nr, self.Nz))

        # Conductivies to build tensor
        self.sigma_H = np.zeros((self.Nr, self.Nz))
        self.sigma_P = np.zeros((self.Nr, self.Nz))
        self.sigma_parallel = np.zeros((self.Nr, self.Nz))

    # TO DO:
    #   This should be part of geometry object, not the hybrid pic model. 
    def compute_volume_field(self, dr, dz, Nr, Nz):
        # Verboncour correction factors for volume calculation
        # Create a 1D array for the radial node control volumes.
        volume = np.empty((Nr,), dtype=np.float64)
        volume[0] = np.pi/3*dr*dr*dz
        if Nr > 1:
            # For nodes 1 through Nr-2.
            i_vals = np.arange(1, Nr-1, dtype=np.float64)
            volume[1:Nr-1] = 2*np.pi*i_vals*dr*dr*dz

        volume[Nr-1] = np.pi*((Nr-1)-1/3)*dr*dr*dz

        # Broadcast the 1D radial volume to a 2D (Nr x Nz) volume field.
        volume_field = np.repeat(volume[:, np.newaxis], Nz, axis=1)
        return volume_field