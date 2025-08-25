import cupy as cp
from centrifugesim import core

print("CuPy device:", cp.cuda.runtime.getDevice())
