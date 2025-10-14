import numpy as np

# Just the Biot Savart solver for Br, Bz that runs once before simulation start

# Should receive coils dictionary with geometry (rc, drc, zc, dzc) and current for each coil

# Then should take that object, solve with Biot-Savart kernels in vacuum_field_solver_kernels
# and fill the fields Br and Bz from hybrid_pic object