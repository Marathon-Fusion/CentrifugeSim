import cupy as cp

BorisPushKernel = cp.RawKernel(r'''
extern "C" __global__
void BorisPushKernel(const int N, const float dt, const float q_m,
                     float* vr, float* vt, float* vz,
                     const float* Ep_r, const float* Ep_t, const float* Ep_z,
                     const float* Bp_r, const float* Bp_t, const float* Bp_z)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        float half_dt = dt * 0.5f;
        
        // Half electric acceleration: v_minus = v + q_m * E * (dt/2)
        float v_minus_r = vr[i] + q_m * Ep_r[i] * half_dt;
        float v_minus_t = vt[i] + q_m * Ep_t[i] * half_dt;
        float v_minus_z = vz[i] + q_m * Ep_z[i] * half_dt;
        
        // Compute rotation vector t = q_m * B * (dt/2)
        float t_r = q_m * Bp_r[i] * half_dt;
        float t_t = q_m * Bp_t[i] * half_dt;
        float t_z = q_m * Bp_z[i] * half_dt;
        
        // Compute squared magnitude of t and the s vector:
        float t2 = t_r * t_r + t_t * t_t + t_z * t_z;
        float denom = 1.0f + t2;
        float s_r = 2.0f * t_r / denom;
        float s_t = 2.0f * t_t / denom;
        float s_z = 2.0f * t_z / denom;
        
        // Compute v_prime = v_minus + cross(v_minus, t)
        float v_prime_r = v_minus_r + (v_minus_t * t_z - v_minus_z * t_t);
        float v_prime_t = v_minus_t + (v_minus_z * t_r - v_minus_r * t_z);
        float v_prime_z = v_minus_z + (v_minus_r * t_t - v_minus_t * t_r);
        
        // Compute v_plus = v_minus + cross(v_prime, s)
        float v_plus_r = v_minus_r + (v_prime_t * s_z - v_prime_z * s_t);
        float v_plus_t = v_minus_t + (v_prime_z * s_r - v_prime_r * s_z);
        float v_plus_z = v_minus_z + (v_prime_r * s_t - v_prime_t * s_r);
        
        // Final half electric acceleration: update the velocities in place.
        vr[i] = v_plus_r + q_m * Ep_r[i] * half_dt;
        vt[i] = v_plus_t + q_m * Ep_t[i] * half_dt;
        vz[i] = v_plus_z + q_m * Ep_z[i] * half_dt;
    }
}
''', 'BorisPushKernel')


gatherEandBKernel = cp.RawKernel(r'''
extern "C" __global__
void gatherEandBKernel(const int N,
                        const int Nr, const int Nz,
                        const float dr, const float dz, const float zmin,
                        const float* r_particles, const float* z_particles,
                        const float* Er, const float* Et, const float* Ez,
                        const float* Br, const float* Bt, const float* Bz,
                        float* Ep_r, float* Ep_t, float* Ep_z,
                        float* Bp_r, float* Bp_t, float* Bp_z)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        // Read particle position
        float r_p = r_particles[idx];
        float z_p = z_particles[idx];
        
        // Compute floating-point indices for r and z.
        float i_float = r_p / dr;
        float j_float = (z_p - zmin) / dz;
        
        // Lower grid indices and interpolation weights.
        int i0 = (int)floorf(i_float);
        int j0 = (int)floorf(j_float);
        float alpha = i_float - i0;
        float beta  = j_float - j0;
        
        // Clamp indices to be within valid range (so that i0+1 and j0+1 are valid).
        if (i0 < 0) i0 = 0;
        if (i0 > Nr - 2) i0 = Nr - 2;
        if (j0 < 0) j0 = 0;
        if (j0 > Nz - 2) j0 = Nz - 2;
        
        // Compute weights for bilinear interpolation.
        float w00 = (1.0f - alpha) * (1.0f - beta);
        float w10 = alpha * (1.0f - beta);
        float w01 = (1.0f - alpha) * beta;
        float w11 = alpha * beta;
                              
        // Compute 1D indices for the four corners in the grid.
        // The grid is stored in row-major order: index = i * Nz + j.
        int idx00 = i0 * Nz + j0;
        int idx10 = (i0 + 1) * Nz + j0;
        int idx01 = i0 * Nz + (j0 + 1);
        int idx11 = (i0 + 1) * Nz + (j0 + 1);
        
        // Bilinearly interpolate the electric field components.
        float interp_Er = w00 * Er[idx00] + w10 * Er[idx10] +
                          w01 * Er[idx01] + w11 * Er[idx11];
        float interp_Et = w00 * Et[idx00] + w10 * Et[idx10] +
                          w01 * Et[idx01] + w11 * Et[idx11];
        float interp_Ez = w00 * Ez[idx00] + w10 * Ez[idx10] +
                          w01 * Ez[idx01] + w11 * Ez[idx11];
        
        // Bilinearly interpolate the magnetic field components.
        float interp_Br = w00 * Br[idx00] + w10 * Br[idx10] +
                          w01 * Br[idx01] + w11 * Br[idx11];
        float interp_Bt = w00 * Bt[idx00] + w10 * Bt[idx10] +
                          w01 * Bt[idx01] + w11 * Bt[idx11];
        float interp_Bz = w00 * Bz[idx00] + w10 * Bz[idx10] +
                          w01 * Bz[idx01] + w11 * Bz[idx11];
        
        // Store the interpolated fields into the particle arrays.
        Ep_r[idx] = interp_Er;
        Ep_t[idx] = interp_Et;
        Ep_z[idx] = interp_Ez;
        
        Bp_r[idx] = interp_Br;
        Bp_t[idx] = interp_Bt;
        Bp_z[idx] = interp_Bz;
    }
}
''', 'gatherEandBKernel')


gatherScalarField = cp.RawKernel(r'''
extern "C" __global__
void gatherScalarField(const int N,
                       const int Nr, const int Nz,
                       const float dr, const float dz, const float zmin,
                       const float* __restrict__ r_particles,
                       const float* __restrict__ z_particles,
                       const float* __restrict__ field,
                       float* __restrict__ Fp)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p < N) {
        // Compute the floating-point grid indices for this particle.
        float i_float = r_particles[p] / dr;
        float j_float = (z_particles[p] - zmin) / dz;
        
        // Lower grid indices.
        int i0 = (int)floorf(i_float);
        int j0 = (int)floorf(j_float);
        
        // Clamp indices to ensure that (i0+1) and (j0+1) are valid.
        if (i0 < 0) i0 = 0;
        if (i0 > Nr - 2) i0 = Nr - 2;
        if (j0 < 0) j0 = 0;
        if (j0 > Nz - 2) j0 = Nz - 2;

        // Fractional parts (weights).
        float alpha = i_float - i0;
        float beta  = j_float - j0;

        // Saturate local coords to [0,1]
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
        beta  = fminf(fmaxf(beta,  0.0f), 1.0f);
        
        // Bilinear weights.
        float w00 = (1.0f - alpha) * (1.0f - beta);
        float w10 =  alpha         * (1.0f - beta);
        float w01 = (1.0f - alpha) * beta;
        float w11 =  alpha         * beta;
                                 
        // Compute 1D indices for the four surrounding grid nodes.
        int idx00 = i0 * Nz + j0;
        int idx10 = (i0 + 1) * Nz + j0;
        int idx01 = i0 * Nz + (j0 + 1);
        int idx11 = (i0 + 1) * Nz + (j0 + 1);

        // Bilinearly interpolate the scalar field.
        float interp_value = w00 * field[idx00] + w10 * field[idx10] +
                             w01 * field[idx01] + w11 * field[idx11];

        // Store the interpolated value in the output array.
        Fp[p] = interp_value;
    }
}
''', "gatherScalarField")


depositScalarKernel = cp.RawKernel(r'''
extern "C" __global__
void depositScalarKernel(const int N,
                           const float* __restrict__ r_particles,
                           const float* __restrict__ z_particles,
                           const float* __restrict__ scalar_p,
                           const float* __restrict__ w_p,
                           double* __restrict__ field,
                           const int Nr, const int Nz,
                           const float dr, const float dz, const float zmin)
{
    int p = blockDim.x * blockIdx.x + threadIdx.x;
    if (p < N) {
        // Compute the floating-point grid indices for this particle.
        float i_float = r_particles[p] / dr;
        float j_float = (z_particles[p] - zmin) / dz;
        
        // Lower grid indices.
        int i0 = (int)floorf(i_float);
        int j0 = (int)floorf(j_float);
        
        // Clamp indices to be within valid range (so that i0+1 and j0+1 are valid).
        if (i0 < 0) i0 = 0;
        if (i0 > Nr - 2) i0 = Nr - 2;
        if (j0 < 0) j0 = 0;
        if (j0 > Nz - 2) j0 = Nz - 2;

        // Fractional parts (weights).
        float alpha = i_float - i0;
        float beta  = j_float - j0;

        // Saturate local coords to [0,1]
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
        beta  = fminf(fmaxf(beta,  0.0f), 1.0f);
                
        // Bilinear weights.
        float w00 = (1.0f - alpha) * (1.0f - beta);
        float w10 =  alpha         * (1.0f - beta);
        float w01 = (1.0f - alpha) * beta;
        float w11 =  alpha         * beta;   
        
        // Compute the particle's contribution (property * weight).
        double deposit = (double)scalar_p[p] * (double)w_p[p];
        
        // Compute 1D indices for the four surrounding grid nodes.
        int idx00 = i0 * Nz + j0;
        int idx10 = (i0 + 1) * Nz + j0;
        int idx01 = i0 * Nz + (j0 + 1);
        int idx11 = (i0 + 1) * Nz + (j0 + 1);
        
        // Deposit contributions atomically.
        atomicAdd(&field[idx00], deposit * w00);
        atomicAdd(&field[idx10], deposit * w10);
        atomicAdd(&field[idx01], deposit * w01);
        atomicAdd(&field[idx11], deposit * w11);
                                   
    }
}
''', "depositScalarKernel")