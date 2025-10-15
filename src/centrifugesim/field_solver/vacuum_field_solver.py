import numpy as np

# Just the Biot Savart solver for Br, Bz that runs once before simulation start

# Should receive coils dictionary with geometry (rc, drc, zc, dzc) and current for each coil

# Then should take that object, solve with Biot-Savart kernels in vacuum_field_solver_kernels
# and fill the fields Br and Bz from hybrid_pic object



# After solving, set the following to hybrid_pic

#hybrid_pic.B_mag = np.sqrt(Br**2 + Bz**2) + 1e-12
#hybrid_pic.br = Br / B_mag
#hybrid_pic.bz = Bz / B_mag