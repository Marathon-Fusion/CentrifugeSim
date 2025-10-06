import numpy as np
import cupy as cp

from centrifugesim import constants

#-------------------------------------------------------------------------------------------------------------------
# everything will be done using operator splitting approach to limit complexity of code
# future versions could include advanced implicit methods to reduce error but will be more computationally expensive
# like Newton krylov method, etc
#-------------------------------------------------------------------------------------------------------------------

# Here function do_step(...) will be called to do 1 single global step evolution of the system.


def do_step():
    
    # Ohms law solver from hybrid_pic_model will be called to initialize phi and E

    # Particle pusher will be called (gather, push, deposit)

    # Then advance neutral gas will also be called (including neutral gas energy equation)

    # Then electron energy equation will be solved (careful here, this should be consistent with neutral gas energy equation)

    # Use proper mask if needed to exclude from the domain the cathode geometry.

    None