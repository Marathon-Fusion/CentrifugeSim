import numpy as np
import cupy as cp

import centrifugesim.constants as constants
from . import particle_container_kernels
from centrifugesim.initialization.init_particles import init_particles_positions_and_weights

class ParticleContainer:

    def __init__(self, Z, m, name, rmax_p, zmin_p, zmax_p, rmax_BC, zmin_BC, zmax_BC, N, Nr, Nz):
        
        self.N = N
        self.Z = Z
        self.q = Z*constants.q_e
        self.m = m
        self.q_m = self.q/m
        self.name = name
        
        self.rmax_p, self.zmin_p, self.zmax_p = rmax_p, zmin_p, zmax_p

        self.rmax_BC = rmax_BC
        self.zmin_BC = zmin_BC
        self.zmax_BC = zmax_BC

        # Allocate arrays for positions (r, z), velocities (v_r, v_t, v_z) and weights
        self.r = None
        self.z = None
        self.vr = None
        self.vt = None
        self.vz = None

        self.weight = None # cp.ones(N).astype(cp.float32)

        # Allocate arrays for particles ids, the id number should be unique to each particle
        # max_prev_id is used for 
        # self.id = cp.ones(N).astype(cp.float32)
        # self.max_prev_id = N

        # Allocate arrays for n, Jr, Jt, Jz, T for this species
        # s stands for species
        self.ns = cp.zeros((Nr, Nz)).astype(cp.float64)
        self.Js_r = cp.zeros((Nr, Nz)).astype(cp.float64)
        self.Js_t = cp.zeros((Nr, Nz)).astype(cp.float64)
        self.Js_z = cp.zeros((Nr, Nz)).astype(cp.float64)
        self.Ts = cp.zeros((Nr, Nz)).astype(cp.float64)

        print(self.name + " initialized")


    def InitParticles(self, r_init, z_init, vr_init, vt_init, vz_init, w_init):

        self.r = cp.copy(r_init).astype(cp.float32)
        self.z = cp.copy(z_init).astype(cp.float32)

        self.vr = cp.copy(vr_init).astype(cp.float32)
        self.vt = cp.copy(vt_init).astype(cp.float32)
        self.vz = cp.copy(vz_init).astype(cp.float32)

        self.weight = cp.copy(w_init).astype(cp.float32)
       

    def BorisPush(self, Ep_r, Ep_t, Ep_z, Bp_r, Bp_t, Bp_z, dt):

        threads_per_block = 512
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        particle_container_kernels.BorisPushKernel((blocks,), (threads_per_block,),
                      (self.N, np.float32(dt), np.float32(self.q_m),
                       self.vr, self.vt, self.vz,
                       Ep_r, Ep_t, Ep_z,
                       Bp_r, Bp_t, Bp_z))
         
      
    def PushX(self, dt):
        
        # Calculate the components of the new position vector
        # (as if in a local Cartesian frame x' = r + vr*dt, y' = vt*dt)
        r_component = self.r + self.vr*dt
        t_component = self.vt*dt

        # Update r and z position
        self.r += self.vr*dt
        self.z += self.vz*dt

        ind_r0 = cp.flatnonzero(self.r<0)
        if(ind_r0.shape[0]>0):
            self.r[ind_r0] = cp.abs(self.r[ind_r0])
            self.vr[ind_r0] = cp.abs(self.vr[ind_r0])
        
        # Calculate the new radial magnitude
        r_new = cp.sqrt(r_component**2 + t_component**2)
        
        # Epsilon for numerical stability to avoid division by zero at r=0
        epsilon = 1e-35
        
        # Calculate sin and cos of the rotation angle
        cos_alpha = r_component / (r_new + epsilon)
        sin_alpha = t_component / (r_new + epsilon)

        # Copy old velocities for rotation
        vr_old = cp.copy(self.vr)
        vt_old = cp.copy(self.vt)
        
        # Apply rotation to get velocity components in the new basis
        self.vr[:] = cos_alpha*vr_old + sin_alpha*vt_old
        self.vt[:] = -sin_alpha*vr_old + cos_alpha*vt_old
                
        self.ApplyBCparticles(self.rmax_p, self.zmin_p, self.zmax_p, dt)


    def ApplyBCparticles(self, rmax, zmin, zmax, dt):
        L = zmax-zmin

        # check particles with r<0
        ind_r0 = cp.flatnonzero(self.r<0)
        if(ind_r0.shape[0]>0):
            self.r[ind_r0] = cp.abs(self.r[ind_r0])
            self.vr[ind_r0] = cp.abs(self.vr[ind_r0])
            #self.vt[ind_r0] = -self.vt[ind_r0] # CHECK

        # check BC at rmax
            # Modify this to check at a given z if a particle is outside of the domain
            # for geometries with variable r(z) BC for particles. Set ID of particle to negative or invalid value
            # then call the particle scrapper routine to remove invalid particles.
            # kernel should be able to interpolate to the z position of particle what is the r(z) of wall
            # and compare with r position of particle.
        ind_rmax = cp.flatnonzero(self.r>rmax)
        if(ind_rmax.shape[0]>0):

            if(self.rmax_BC=="reflecting"):
                self.r[ind_rmax] = rmax
                self.vr[ind_rmax] = -self.vr[ind_rmax]

            elif(self.rmax_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_rmax)


        # check BC at zmin
        ind_zmin = cp.flatnonzero(self.z<zmin)
        if(ind_zmin.shape[0]>0):

            if(self.zmin_BC=="reflecting"):
                self.z[ind_zmin] = zmin
                self.vz[ind_zmin] = -self.vz[ind_zmin]

            elif(self.zmin_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_zmin)

            elif(self.zmin_BC=="periodic"):
                self.z[ind_zmin] += L


        # check BC at zmax
        ind_zmax = cp.flatnonzero(self.z>zmax)
        if(ind_zmax.shape[0]>0):

            if(self.zmax_BC=="reflecting"):
                self.z[ind_zmax] = zmax
                self.vz[ind_zmax] = -self.vz[ind_zmax]

            elif(self.zmax_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_zmax)

            elif(self.zmax_BC=="periodic"):
                self.z[ind_zmax] -= L


    def gatherEandB(self, Er, Et, Ez, Br, Bt, Bz, dr, dz, zmin):

        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        Nr = Er.shape[0]
        Nz = Er.shape[1]

        Ep_r, Ep_t, Ep_z = (cp.zeros_like(self.r) for _ in range(3))
        Bp_r, Bp_t, Bp_z = (cp.zeros_like(self.r) for _ in range(3))

        particle_container_kernels.gatherEandBKernel((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Er.astype(cp.float32), Et.astype(cp.float32), Ez.astype(cp.float32), 
                        Br.astype(cp.float32), Bt.astype(cp.float32), Bz.astype(cp.float32),
                        Ep_r, Ep_t, Ep_z, Bp_r, Bp_t, Bp_z))
        
        return Ep_r, Ep_t, Ep_z, Bp_r, Bp_t, Bp_z
    

    def gatherEandB_v2(self, Er, Et, Ez, Br, Bt, Bz, dr, dz, zmin):

        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        Nr = Er.shape[0]
        Nz = Er.shape[1]

        Ep_r, Ep_t, Ep_z = (cp.zeros_like(self.r) for _ in range(3))
        Bp_r, Bp_t, Bp_z = (cp.zeros_like(self.r) for _ in range(3))

        # Gather E
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Er.astype(cp.float32), Ep_r))
        
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Et.astype(cp.float32), Ep_t))
        
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Ez.astype(cp.float32), Ep_z))
        
        # Gather B
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Br.astype(cp.float32), Bp_r))
        
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Bt.astype(cp.float32), Bp_t))
        
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Bz.astype(cp.float32), Bp_z))
        
        return Ep_r, Ep_t, Ep_z, Bp_r, Bp_t, Bp_z
        

    def gatherScalarField(self, field, dr, dz, zmin):

        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        Nr = field.shape[0]
        Nz = field.shape[1]

        Fp = cp.zeros(self.N).astype(cp.float32)

        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                (self.N, Nr, Nz, 
                np.float32(dr), np.float32(dz), np.float32(zmin),
                self.r, self.z,
                field, Fp))
        
        cp.cuda.Stream.null.synchronize()
        return Fp


    def depositScalarField(self, field, Fp, dr, dz, zmin):

        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        Nr = field.shape[0]
        Nz = field.shape[1]

        field*=0
        particle_container_kernels.depositScalarKernel((blocks,), (threads_per_block,),
                (self.N, self.r, self.z, Fp, self.weight, field,
                 Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin) ))
        
        cp.cuda.Stream.null.synchronize()
        return field


    def depositNandJ(self, volume_field, dr, dz, zmin):
        Nr, Nz = volume_field.shape

        volume_field_d = cp.asarray(volume_field)

        self.ns = self.depositScalarField(self.ns, cp.ones_like(self.r).astype(cp.float32), dr, dz, zmin)
        self.ns = self.ns/volume_field_d

        self.Js_r = self.depositScalarField(self.Js_r, (self.vr*self.q).astype(cp.float32), dr, dz, zmin)
        self.Js_r = self.Js_r/volume_field_d

        self.Js_t = self.depositScalarField(self.Js_t, (self.vt*self.q).astype(cp.float32), dr, dz, zmin)
        self.Js_t = self.Js_t/volume_field_d

        self.Js_z = self.depositScalarField(self.Js_z, (self.vz*self.q).astype(cp.float32), dr, dz, zmin)
        self.Js_z = self.Js_z/volume_field_d

        # Write apply_BC_Js and call here for each component.
        self.Js_r[0,:] = 0
        self.Js_t[0,:] = 0
        self.Js_z[0,:] = self.Js_z[1,:]

        self.ns[:,0] = self.ns[:,1]
        self.ns[:,Nz-1] = self.ns[:,Nz-2]

        self.Js_r[:,0] = self.Js_r[:,1]
        self.Js_r[:,Nz-1] = self.Js_r[:,Nz-2]

        self.Js_t[:,0] = self.Js_t[:,1]
        self.Js_t[:,Nz-1] = self.Js_t[:,Nz-2]

        self.Js_z[:,0] = self.Js_z[:,1]
        self.Js_z[:,Nz-1] = self.Js_z[:,Nz-2]

        return self.ns, self.Js_r, self.Js_t, self.Js_z


    def depositTemperature_dim(self, u_field, vp, dr, dz, zmin, Nr, Nz):

        Ts_dim = cp.zeros((Nr, Nz)).astype(cp.float64)
        n_PPC = cp.zeros((Nr, Nz)).astype(cp.float64)

        # Gather from u_field to particles positions to calculate (u-v)
        up = self.gatherScalarField(u_field.astype(cp.float32), dr, dz, zmin)

        # Now calculate and deposit variance wp^2 = (up-vp)^2
        wp2 = (1.0/self.weight)*(up-vp)**2
        
        self.depositScalarField(Ts_dim, wp2, dr, dz, zmin)
        self.depositScalarField(n_PPC, 1.0/self.weight, dr, dz, zmin)
        Ts_dim = self.m*Ts_dim/constants.kb
        Ts_dim = cp.where(n_PPC>0,Ts_dim/n_PPC,0)

        return Ts_dim
    

    # Should try doing this in a kernel (with 1 single pass to calculate everything)
    # Look at Eric's implementation of variance deposition in warpx!
    def depositTemperature(self, n_floor, dr, dz, zmin, Nr, Nz):
        rho_limited = cp.where(self.ns<n_floor, n_floor, self.ns)*self.q
        ur = (self.Js_r/rho_limited).astype(cp.float32)
        ut = (self.Js_t/rho_limited).astype(cp.float32)
        uz = (self.Js_z/rho_limited).astype(cp.float32)

        Ts_r = self.depositTemperature_dim(ur, self.vr, dr, dz, zmin, Nr, Nz)
        Ts_t = self.depositTemperature_dim(ut, self.vt, dr, dz, zmin, Nr, Nz)
        Ts_z = self.depositTemperature_dim(uz, self.vz, dr, dz, zmin, Nr, Nz)

        self.Ts = (Ts_r + Ts_t + Ts_z) / 3.0

    def remove_invalid_particles(self):
        ind_invalid = cp.flatnonzero(self.id<0)
        if(ind_invalid.shape[0]>0):
            self.remove_indices_and_free_memory(ind_invalid)

    def remove_indices_and_free_memory(self, indices_to_remove):
        # Create a boolean mask where True means "keep this element"
        mask = cp.ones(self.r.size, dtype=bool)
        mask[indices_to_remove] = False
        
        # Apply the mask to get a new array
        self.r = cp.copy(self.r[mask])
        self.z = cp.copy(self.z[mask])

        self.vr = cp.copy(self.vr[mask])
        self.vt = cp.copy(self.vt[mask])
        self.vz = cp.copy(self.vz[mask])

        self.weight = cp.copy(self.weight[mask])

        self.N = self.N - indices_to_remove.shape[0]
        # Free unused memory blocks
        cp._default_memory_pool.free_all_blocks()

    """
    # Update this ! currently calling in cpu not here
    def apply_bilinear_filter(self):
        self.ns = bilinear_filter(self.ns)
        self.Js_r = bilinear_filter(self.Js_r)
        self.Js_t = bilinear_filter(self.Js_t)
        self.Js_z = bilinear_filter(self.Js_z)
    """

    def copy_fields_to_host(self):
        ns = cp.asnumpy(self.ns)
        Js_r = cp.asnumpy(self.Js_r)
        Js_t = cp.asnumpy(self.Js_t)
        Js_z = cp.asnumpy(self.Js_z)
        return ns, Js_r, Js_t, Js_z
    
    # TO DO: Add ids too ! so I can do particle scrapper calls to remove particles with negative id.
    def add_N_particles(self, 
                        r_new, z_new, 
                        vr_new, vt_new, vz_new,
                        w_new ):
        if(self.N is None or self.N==0):
            self.r = r_new
            self.z = z_new
            self.vr = vr_new
            self.vt = vt_new
            self.vz = vz_new
            self.weight = w_new
            self.N+=self.r.shape[0]
            
        else:
            self.r = cp.concatenate((self.r, r_new)).astype(cp.float32)
            self.z = cp.concatenate((self.z, z_new)).astype(cp.float32)
            self.vr = cp.concatenate((self.vr, vr_new)).astype(cp.float32)
            self.vt = cp.concatenate((self.vt, vt_new)).astype(cp.float32)
            self.vz = cp.concatenate((self.vz, vz_new)).astype(cp.float32)
            self.weight = cp.concatenate((self.weight, w_new)).astype(cp.float32)
            self.N=self.r.shape[0]

    def add_plasma_uniform_density(
        self,
        nr: int,
        r_min: float,
        r_max: float,
        z_min: float,
        z_max: float,
        dr: float,
        dz: float,
        density: float,
        vth_i: float,
        w: int = None,
        np_per_cell: int = None,
        seed: int = None,
    ) -> None:
        """
        Add plasma particles with uniform density to the simulation.

        Parameters
        ----------
        self : object
            Simulation instance.
        np_per_cell : int
            Number of particles per cell.
        weight : int
            weight of macroparticles
        r_min, r_max : float
            Radial bounds for particle addition.
        z_min, z_max : float
            Axial (z) bounds for particle addition.
        dr, dz : float
            Grid spacing in the radial and axial directions.
        density : float
            Physical particle density (particles per unit volume).
        vth_i : float
            Ion thermal velocity for sampling.
        seed : int, optional
            Seed for reproducibility.
        """
        Np, r_pos, z_pos, weights = init_particles_positions_and_weights(
            nr,
            r_min,
            r_max,
            z_min,
            z_max,
            dr,
            dz,
            density,
            w,
            np_per_cell,
            seed,
        )

        # Transfer to GPU and sample velocities
        r_new = cp.asarray(r_pos, dtype=cp.float32)
        z_new = cp.asarray(z_pos, dtype=cp.float32)
        vr_new = cp.random.normal(0, vth_i, size=Np).astype(cp.float32)
        vt_new = cp.random.normal(0, vth_i, size=Np).astype(cp.float32)
        vz_new = cp.random.normal(0, vth_i, size=Np).astype(cp.float32)
        w_new = cp.asarray(weights, dtype=cp.float32)

        self.add_N_particles(
            r_new=r_new,
            z_new=z_new,
            vr_new=vr_new,
            vt_new=vt_new,
            vz_new=vz_new,
            w_new=w_new,
        )

    def add_from_numpy_arrays(self,
                              w_p,
                              r_p, z_p,
                              vr_p, vt_p, vz_p):

        r_new = cp.asarray(r_p, dtype=cp.float32)
        z_new = cp.asarray(z_p, dtype=cp.float32)
        vr_new = cp.asarray(vr_p, dtype=cp.float32)
        vt_new = cp.asarray(vt_p, dtype=cp.float32)
        vz_new = cp.asarray(vz_p, dtype=cp.float32)
        w_new = cp.asarray(w_p, dtype=cp.float32)

        self.add_N_particles(
            r_new=r_new,
            z_new=z_new,
            vr_new=vr_new,
            vt_new=vt_new,
            vz_new=vz_new,
            w_new=w_new,
        )

        self.ApplyBCparticles(self.rmax_p, self.zmin_p, self.zmax_p, 0)

    def drag_diffusion(self,
                       Uer, Uet, Uez,
                       ne, Te, 
                       nu_ei,
                       n_floor,
                       dr, dz,
                       zmin,
                       dt):
        """
        Uer, Uet, Uez, ne, Te and nu_ei should already be float32 device arrays (Nr, Nz)
        """
        # Gather each field to ions positions
        ne_p =  self.gatherScalarField(ne.astype(cp.float32), dr, dz, zmin)
        Te_p =  self.gatherScalarField(Te.astype(cp.float32), dr, dz, zmin)
        nu_ei_p =  self.gatherScalarField(nu_ei.astype(cp.float32), dr, dz, zmin)
        uer_p =  self.gatherScalarField(Uer.astype(cp.float32), dr, dz, zmin)
        uet_p =  self.gatherScalarField(Uet.astype(cp.float32), dr, dz, zmin)
        uez_p =  self.gatherScalarField(Uez.astype(cp.float32), dr, dz, zmin)

        ind = cp.flatnonzero(ne_p>n_floor)

        if(ind.shape[0]>0):

            nu_drag_p = nu_ei_p*constants.m_e/self.m
            D_p = nu_drag_p*constants.kb*Te_p/self.m
            diffusion_term_p = cp.sqrt(2*D_p*dt)

            R = cp.random.randn(self.N, 3)

            dvr_ = (- nu_drag_p[ind]*(self.vr[ind]-uer_p[ind])*dt + diffusion_term_p[ind]*R[ind,0]).astype(cp.float32)
            dvt_ = (- nu_drag_p[ind]*(self.vt[ind]-uet_p[ind])*dt + diffusion_term_p[ind]*R[ind,1]).astype(cp.float32)
            dvz_ = (- nu_drag_p[ind]*(self.vz[ind]-uez_p[ind])*dt + diffusion_term_p[ind]*R[ind,2]).astype(cp.float32)

            self.vr[ind] += dvr_
            self.vt[ind] += dvt_
            self.vz[ind] += dvz_