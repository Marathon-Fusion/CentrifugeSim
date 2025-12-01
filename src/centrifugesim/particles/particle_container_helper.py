import numpy as np
from numba import njit, prange

@njit(parallel=False)
def cic_deposit_renormalized(p_r, p_z, p_w, mask, dr, dz, r_min, z_min):
    """
    Numba-accelerated Cloud-in-Cell deposition with Mass Conservation (Renormalization).
    
    If a particle overlaps a solid boundary (mask=0), its weight is redistributed 
    proportionally to the remaining valid nodes (mask=1) it touches.
    """
    Nr, Nz = mask.shape
    density_accum = np.zeros((Nr, Nz), dtype=np.float64)
    
    n_particles = len(p_r)
    
    for k in range(n_particles):
        r_pos = p_r[k]
        z_pos = p_z[k]
        total_particle_weight = p_w[k]
        
        # 1. Normalize coordinates
        r_norm = (r_pos - r_min) / dr
        z_norm = (z_pos - z_min) / dz
        
        # 2. Find lower-left node index
        ir = int(r_norm)
        iz = int(z_norm)
        
        # 3. Bilinear weights (Geometric overlap)
        hr = r_norm - ir
        hz = z_norm - iz
        
        w00 = (1.0 - hr) * (1.0 - hz)
        w10 = hr * (1.0 - hz)
        w01 = (1.0 - hr) * hz
        w11 = hr * hz
        
        # 4. Check Validity of the 4 surrounding nodes
        # We need to see which nodes are legally allowed to accept density
        
        valid00 = (ir >= 0 and ir < Nr and iz >= 0 and iz < Nz) and (mask[ir, iz] == 1)
        valid10 = (ir + 1 >= 0 and ir + 1 < Nr and iz >= 0 and iz < Nz) and (mask[ir + 1, iz] == 1)
        valid01 = (ir >= 0 and ir < Nr and iz + 1 >= 0 and iz + 1 < Nz) and (mask[ir, iz + 1] == 1)
        valid11 = (ir + 1 >= 0 and ir + 1 < Nr and iz + 1 >= 0 and iz + 1 < Nz) and (mask[ir + 1, iz + 1] == 1)
        
        # 5. Calculate Sum of Valid Weights
        sum_valid = 0.0
        if valid00: sum_valid += w00
        if valid10: sum_valid += w10
        if valid01: sum_valid += w01
        if valid11: sum_valid += w11
        
        # 6. Renormalize and Deposit
        # If the particle touches at least one valid plasma node:
        if sum_valid > 1e-12:
            # Scaling factor to ensure conservation of mass
            # All the weight that WOULD have gone to the wall is distributed to the plasma
            scale = total_particle_weight / sum_valid
            
            if valid00: density_accum[ir, iz]         += w00 * scale
            if valid10: density_accum[ir + 1, iz]     += w10 * scale
            if valid01: density_accum[ir, iz + 1]     += w01 * scale
            if valid11: density_accum[ir + 1, iz + 1] += w11 * scale
            
    return density_accum

#################################################################################
############################# RZ Bilinear Smoothing #############################
#################################################################################

@njit(parallel=True, fastmath=True)
def rz_bilinear_smooth(ns, mask, r_coords, nr, nz):
    """
    Applies an RZ-compatible bilinear filter to density ns.
    Method: Separable convolution with Volume Weighting.
    """
    # 1. Allocate buffers
    ns_z_smoothed = np.zeros_like(ns)
    ns_final = np.zeros_like(ns)
    
    # Kernel weights (3-point)
    w = np.array([0.25, 0.5, 0.25])
    
    # =========================================================
    # PASS 1: Z-Direction (Axial)
    # Volume is constant along Z lines, so standard smoothing applies.
    # =========================================================
    for i in prange(nr):
        for j in range(nz):
            if mask[i, j] == 0:
                continue

            num = 0.0
            den = 0.0
            
            # Iterate kernel: j-1, j, j+1
            for k in range(-1, 2):
                neighbor_j = j + k
                
                # Check Bounds
                if neighbor_j >= 0 and neighbor_j < nz:
                    # Check Mask
                    if mask[i, neighbor_j] == 1:
                        num += ns[i, neighbor_j] * w[k+1]
                        den += w[k+1]
            
            if den > 1e-12:
                ns_z_smoothed[i, j] = num / den
            else:
                ns_z_smoothed[i, j] = ns[i, j]

    # =========================================================
    # PASS 2: R-Direction (Radial)
    # Volume varies with R. We must weight by Radius.
    # Formula: Sum(n * w * r) / Sum(w * r)
    # =========================================================
    dr = r_coords[1] - r_coords[0]
    
    # Effective radius for the axis node (r=0) to prevent weight collapse
    # We treat the axis node as representing the volume centroid of the first cell
    r_axis_eff = dr * 0.25

    for j in prange(nz):
        for i in range(nr):
            if mask[i, j] == 0:
                continue

            num = 0.0
            den = 0.0
            
            for k in range(-1, 2):
                neighbor_i = i + k
                
                if neighbor_i >= 0 and neighbor_i < nr:
                    if mask[neighbor_i, j] == 1:
                        # Determine Geometric Weight (Radius)
                        r_neighbor = r_coords[neighbor_i]
                        
                        # Handle Axis Singularity
                        if r_neighbor < 1e-12:
                            r_val = r_axis_eff
                        else:
                            r_val = r_neighbor
                        
                        # Apply Weight: Kernel * Geometry
                        weight = w[k+1] * r_val
                        
                        num += ns_z_smoothed[neighbor_i, j] * weight
                        den += weight
            
            if den > 1e-12:
                ns_final[i, j] = num / den
            else:
                ns_final[i, j] = ns_z_smoothed[i, j]

    return ns_final