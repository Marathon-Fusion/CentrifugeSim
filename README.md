# CentrifugeSim
Plasma centrifuge Hybrid-PIC code (kinetic ions, fluid electrons, fluid neutrals).
First version of code, particle algorithm runs on cuda-c. Rest on cpu using simple python/numba functions. Will move to gpu once grid get's larger to not kill performance with kernel launch overhead.
