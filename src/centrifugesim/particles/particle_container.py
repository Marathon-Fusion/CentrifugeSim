import numpy as np
import cupy as cp

import centrifugesim.constants as constants
from . import particle_container_kernels

from centrifugesim.initialization.init_particles import initialize_ions_from_ni_nppc

class ParticleContainer:

    def __init__(self, Z, m, name, rmax_BC, zmin_BC, zmax_BC, N, geom):
        
        self.N = N
        self.Z = Z
        self.q = Z*constants.q_e
        self.m = m
        self.q_m = self.q/m
        self.name = name
        
        self.rmax_p, self.zmin_p, self.zmax_p = geom.rmax, geom.zmin, geom.zmax

        self.rmax_BC = rmax_BC
        self.zmin_BC = zmin_BC
        self.zmax_BC = zmax_BC

        # Allocate arrays for positions (r, z), velocities (v_r, v_t, v_z) and weights
        self.r = None
        self.z = None
        self.vr = None
        self.vt = None
        self.vz = None

        self.weight = None

        # Allocate arrays for n, Jr, Jt, Jz, T for this species
        # s stands for species
        self.ns = cp.zeros((geom.Nr, geom.Nz)).astype(cp.float64)
        self.Js_r = cp.zeros((geom.Nr, geom.Nz)).astype(cp.float64)
        self.Js_t = cp.zeros((geom.Nr, geom.Nz)).astype(cp.float64)
        self.Js_z = cp.zeros((geom.Nr, geom.Nz)).astype(cp.float64)
        self.Ts = cp.zeros((geom.Nr, geom.Nz)).astype(cp.float64)

        self.us_r_host = np.zeros((geom.Nr, geom.Nz)).astype(np.float64)
        self.us_t_host = np.zeros((geom.Nr, geom.Nz)).astype(np.float64)
        self.us_z_host = np.zeros((geom.Nr, geom.Nz)).astype(np.float64)

        print(self.name + " initialized")


    def InitParticles(self, r_init, z_init, vr_init, vt_init, vz_init, w_init):

        self.r = cp.copy(r_init).astype(cp.float32)
        self.z = cp.copy(z_init).astype(cp.float32)

        self.vr = cp.copy(vr_init).astype(cp.float32)
        self.vt = cp.copy(vt_init).astype(cp.float32)
        self.vz = cp.copy(vz_init).astype(cp.float32)

        self.weight = cp.copy(w_init).astype(cp.float32)
       
    def compute_u(self, n_floor):
        rho_s_host = cp.asnumpy(self.ns)*self.q
        rho_floor = n_floor*self.q
        Js_r_host = cp.asnumpy(self.Js_r)
        Js_t_host = cp.asnumpy(self.Js_t)
        Js_z_host = cp.asnumpy(self.Js_z)

        self.us_r_host = np.where(rho_s_host<rho_floor, 0, Js_r_host/rho_s_host)
        self.us_t_host = np.where(rho_s_host<rho_floor, 0, Js_t_host/rho_s_host)
        self.us_z_host = np.where(rho_s_host<rho_floor, 0, Js_z_host/rho_s_host)

    def BorisPush(self, Ep_r, Ep_t, Ep_z, Bp_r, Bp_t, Bp_z, dt):

        threads_per_block = 512
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        particle_container_kernels.BorisPushKernel((blocks,), (threads_per_block,),
                      (self.N, np.float32(dt), np.float32(self.q_m),
                       self.vr, self.vt, self.vz,
                       Ep_r, Ep_t, Ep_z,
                       Bp_r, Bp_t, Bp_z))
         
      
    def PushX(self, geom, dt):
        
        # Calculate the components of the new position vector
        # (as if in a local Cartesian frame x' = r + vr*dt, y' = vt*dt)
        r_component = self.r + self.vr*dt
        t_component = self.vt*dt

        # Update r and z position
        self.z += self.vz*dt
        
        # Calculate the new radial magnitude
        r_new = cp.sqrt(r_component**2 + t_component**2)
                
        # Calculate sin and cos of the rotation angle
        cos_alpha = r_component / r_new 
        sin_alpha = t_component / r_new

        # Copy old velocities for rotation
        vr_old = cp.copy(self.vr)
        vt_old = cp.copy(self.vt)
        
        # Apply rotation to get velocity components in the new basis
        self.vr[:] = cos_alpha*vr_old + sin_alpha*vt_old
        self.vt[:] = -sin_alpha*vr_old + cos_alpha*vt_old

        self.r[:] = r_new
        
        self.ApplyBCparticles(geom, self.rmax_p, self.zmin_p, self.zmax_p)


    def ApplyBCparticles(self, geom, rmax, zmin, zmax):
        L = zmax-zmin

        # check particles with r<0
        ind_r0 = cp.flatnonzero(self.r<0)
        if(ind_r0.shape[0]>0):
            self.r[ind_r0] = cp.abs(self.r[ind_r0])
            self.vr[ind_r0] = -self.vr[ind_r0]
            self.vt[ind_r0] = -self.vt[ind_r0]

        # check particles with r>rmax
        ind_rmax = cp.flatnonzero(self.r>rmax)
        if(ind_rmax.shape[0]>0):

            if(self.rmax_BC=="reflecting"):
                self.r[ind_rmax]  = 2.0*rmax - self.r[ind_rmax]
                self.vr[ind_rmax] = -self.vr[ind_rmax]

            elif(self.rmax_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_rmax)


        # check BC at zmin
        ind_zmin = cp.flatnonzero(self.z<zmin)
        if(ind_zmin.shape[0]>0):

            if(self.zmin_BC=="reflecting"):
                self.z[ind_zmin] = zmin
                self.vz[ind_zmin] = cp.abs(self.vz[ind_zmin])

            elif(self.zmin_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_zmin)

            elif(self.zmin_BC=="periodic"):
                self.z[ind_zmin] += L


        # check BC at zmax
        ind_zmax = cp.flatnonzero(self.z>zmax)
        if(ind_zmax.shape[0]>0):

            if(self.zmax_BC=="reflecting"):
                self.z[ind_zmax] = zmax
                self.vz[ind_zmax] = -cp.abs(self.vz[ind_zmax])

            elif(self.zmax_BC=="absorbing"):
                self.remove_indices_and_free_memory(ind_zmax)

            elif(self.zmax_BC=="periodic"):
                self.z[ind_zmax] -= L

        # check if particles are asorbed at cathode:
        ind_cathode = cp.flatnonzero(cp.logical_and(self.r<=geom.rmax_cathode,self.z<=geom.zmax_cathode))
        if(ind_cathode.shape[0]>0):
            self.remove_indices_and_free_memory(ind_cathode)

        # check if particles are absorbed at first anode:
        ind_anode_1 = cp.flatnonzero(cp.logical_and(self.r>=geom.rmin_anode,
                        cp.logical_and(self.z>=geom.zmin_anode,self.z<=geom.zmax_anode)))
        if(ind_anode_1.shape[0]>0):
            self.remove_indices_and_free_memory(ind_anode_1)

        # check if particles are absorbed at second anode:
        dz_anode = geom.zmax_anode - geom.zmin_anode
        ind_anode_2 = cp.flatnonzero(cp.logical_and(self.r>=geom.rmin_anode,
                        cp.logical_and(self.z>=geom.zmin_anode2,self.z<=geom.zmin_anode2+dz_anode)))
        if(ind_anode_2.shape[0]>0):
            self.remove_indices_and_free_memory(ind_anode_2)


    def gatherEandB(self, Er, Et, Ez, Br, Bt, Bz, geom):
        """
        Gather function from cartesian rz mesh to particles positions.
        Only gathering r and z components of E and B fields
        """
        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        Nr = Er.shape[0]
        Nz = Er.shape[1]

        dr = geom.dr
        dz = geom.dz
        zmin = geom.zmin

        Ep_r, Ep_t, Ep_z = (cp.zeros_like(self.r) for _ in range(3))
        Bp_r, Bp_t, Bp_z = (cp.zeros_like(self.r) for _ in range(3))

        # Gather E
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Er.astype(cp.float32), Ep_r))
                
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Ez.astype(cp.float32), Ep_z))
        
        # Gather B
        particle_container_kernels.gatherScalarField((blocks,), (threads_per_block,),
                       (self.N, Nr, Nz, np.float32(dr), np.float32(dz), np.float32(zmin), self.r, self.z,
                        Br.astype(cp.float32), Bp_r))
            
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


    def depositNandJ(self, geom):
        Nr, Nz = geom.Nr, geom.Nz
        dr, dz, zmin = geom.dr, geom.dz, geom.zmin
        volume_field = geom.volume_field

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

        # change to verboncour
        self.ns[0,:] = self.ns[1,:]
        self.ns[Nr-1,:] = self.ns[Nr-2,:]

        self.Js_r[:,0] = self.Js_r[:,1]
        self.Js_r[:,Nz-1] = self.Js_r[:,Nz-2]

        self.Js_t[:,0] = self.Js_t[:,1]
        self.Js_t[:,Nz-1] = self.Js_t[:,Nz-2]

        self.Js_z[:,0] = self.Js_z[:,1]
        self.Js_z[:,Nz-1] = self.Js_z[:,Nz-2]

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
    
    def depositTemperature(self, n_floor, geom):
        dr, dz, zmin, Nr, Nz = geom.dr, geom.dz, geom.zmin, geom.Nr, geom.Nz

        rho_limited = cp.where(self.ns<n_floor, n_floor, self.ns)*self.q
        ur = (self.Js_r/rho_limited).astype(cp.float32)
        ut = (self.Js_t/rho_limited).astype(cp.float32)
        uz = (self.Js_z/rho_limited).astype(cp.float32)

        Ts_r = self.depositTemperature_dim(ur, self.vr, dr, dz, zmin, Nr, Nz)
        Ts_t = self.depositTemperature_dim(ut, self.vt, dr, dz, zmin, Nr, Nz)
        Ts_z = self.depositTemperature_dim(uz, self.vz, dr, dz, zmin, Nr, Nz)

        self.Ts = (Ts_r + Ts_t + Ts_z) / 3.0
        self.Ts = cp.where(self.ns<n_floor, 0, self.Ts)

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

    def copy_fields_to_host(self):
        ns = cp.asnumpy(self.ns)
        Js_r = cp.asnumpy(self.Js_r)
        Js_t = cp.asnumpy(self.Js_t)
        Js_z = cp.asnumpy(self.Js_z)
        return ns, Js_r, Js_t, Js_z
    
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

    def add_from_numpy_arrays(self,
                              geom,
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

        self.ApplyBCparticles(geom, self.rmax_p, self.zmin_p, self.zmax_p)

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

    def reset_particles(self, geom, nppc):
        """
        This function removes all the ions in the domain and create new ones 
        recreating the n, T, ur, ut, uz fields with a given new nppc.
        This should not be used often since it forces the IVDF to be Maxwellian (+ some pre existing drift)
        which is equivalent to adding a collision operator with a frequency given by 1/dt_call where
        dt_call is the time interval between consecutive reset_particles calls.
        A particle merge algorithm should be used instead if particles are expected to have non Maxwellian distributions
        but it would increase the computational cost.
        """
        # Remove all particles
        self.remove_indices_and_free_memory(cp.arange(self.N))

        # Create new ones with set nppc and last deposited fields
        ns, Js_r, Js_t, Js_z = self.copy_fields_to_host()
        Ts = cp.asnumpy(self.Ts)

        # create numpy arrays for new particles from ni, Ti fields with set nppc
        w_p, r_p, z_p, vr_p, vt_p, vz_p = initialize_ions_from_ni_nppc(ns, Ts, geom.R, geom.Z, geom.dr, geom.dz, nppc, self.m)

        # add particles to container
        self.add_from_numpy_arrays(geom, w_p, r_p, z_p, vr_p, vt_p, vz_p)

        # Add drift velocities to particles
        uir_grid_device = cp.asarray(self.us_r_host).astype(cp.float32)
        uit_grid_device = cp.asarray(self.us_t_host).astype(cp.float32)
        uiz_grid_device = cp.asarray(self.us_z_host).astype(cp.float32)
        ui_r =  self.gatherScalarField(uir_grid_device, geom.dr, geom.dz, geom.zmin)
        ui_t =  self.gatherScalarField(uit_grid_device, geom.dr, geom.dz, geom.zmin)
        ui_z =  self.gatherScalarField(uiz_grid_device, geom.dr, geom.dz, geom.zmin)
        self.vr += ui_r.astype(cp.float32)
        self.vt += ui_t.astype(cp.float32)
        self.vz += ui_z.astype(cp.float32)

        del uir_grid_device, uit_grid_device, uiz_grid_device
        del ui_r, ui_t, ui_z
        cp._default_memory_pool.free_all_blocks()


    def collide_with_neutrals_mcc(
        self,
        geom,
        Tn_host, nn_host,
        un_r_host, un_t_host, un_z_host,
        m_n,
        dt,
        sigma_mt=5e-19,
        seed=None
    ):
        """
        Monte-Carlo collisions with background neutrals.

        Parameters
        ----------
        Tn_host : np.ndarray, shape (Nr,Nz), float64
            Neutral temperature field [K] on host.
        nn_host : np.ndarray, shape (Nr,Nz), float64
            Neutral number density field [m^-3] on host.
        un_r_host, un_theta_host, un_z_host : np.ndarray (Nr,Nz), float64
            Neutral bulk velocity components [m/s] on host, cylindrical components.
        m_n : float
            Neutral mass [kg].
        sigma_mt : float, optional
            Momentum-transfer cross-section [m^2]. Default 5e-19.
        seed : int or None
            RNG seed for reproducibility of collision sampling.

        Notes
        -----
        * Uses P_coll = 1 - exp(-nn * sigma_mt * u * dt) with u = |v_i - v_n|.
        * Samples one neutral velocity from a Maxwellian at local Tn and bulk (u_n) per ion.
        * Performs isotropic scattering of the relative velocity.
        * Ion velocity update: v_i <- v_i + (m_n/(m_i+m_n)) * Δu    (matches 1/2 for Ar^+–Ar).
        """
        # check number of ions
        N = self.N
        if N == 0:
            return  # nothing to do

        m_i = float(self.m)

        Tn_device = cp.asarray(Tn_host).astype(cp.float32)
        nn_device = cp.asarray(nn_host).astype(cp.float32)
        un_r_device = cp.asarray(un_r_host).astype(cp.float32)
        un_t_device = cp.asarray(un_t_host).astype(cp.float32)
        un_z_device = cp.asarray(un_z_host).astype(cp.float32)

        # Gather local neutral state at particle positions
        Tn_loc = self.gatherScalarField(Tn_device, geom.dr, geom.dz, geom.zmin)     # [K]      
        nn_loc = self.gatherScalarField(nn_device, geom.dr, geom.dz, geom.zmin)     # [m^-3]
        unr_loc = self.gatherScalarField(un_r_device, geom.dr, geom.dz, geom.zmin)  # [m/s]
        unt_loc = self.gatherScalarField(un_t_device, geom.dr, geom.dz, geom.zmin)  # [m/s]
        unz_loc = self.gatherScalarField(un_z_device, geom.dr, geom.dz, geom.zmin)  # [m/s]

        # Output buffers (in-place is fine, but separate arrays keep it safe)
        vr_out = cp.empty_like(self.vr)
        vt_out = cp.empty_like(self.vt)
        vz_out = cp.empty_like(self.vz)

        # Random numbers: generate with NumPy then move to CuPy (as requested)
        rcoll = cp.random.rand(N, dtype=cp.float32)     # collision acceptance
        rbeta = cp.random.rand(N, dtype=cp.float32)     # for cos(beta) = 1 - 2 R1
        rphi  = cp.random.rand(N, dtype=cp.float32)     # for phi = 2*pi*R2
        g1    = cp.random.randn(N, dtype=cp.float32)    # Gaussian for neutral thermal vx
        g2    = cp.random.randn(N, dtype=cp.float32)    # Gaussian for neutral thermal vy
        g3    = cp.random.randn(N, dtype=cp.float32)    # Gaussian for neutral thermal vz

        # Launch kernel
        threads_per_block = 256
        blocks = (self.N + threads_per_block - 1) // threads_per_block

        particle_container_kernels.mccKernel(
            (blocks,), (threads_per_block,),
            (
                self.vr, self.vt, self.vz,
                unr_loc, unt_loc, unz_loc,
                nn_loc, Tn_loc,
                rcoll, rbeta, rphi,
                g1, g2, g3,
                vr_out, vt_out, vz_out,
                np.float64(dt),
                np.float64(sigma_mt),
                np.float64(m_i),
                np.float64(m_n),
                np.float64(constants.kb),
                np.int64(N),
            )
        )

        # Write back updated velocities
        self.vr[...] = vr_out
        self.vt[...] = vt_out
        self.vz[...] = vz_out

        # del and free memory
        del vr_out, vt_out, vz_out
        del g1, g2, g3
        del rcoll, rbeta, rphi
        del Tn_loc, nn_loc, unr_loc, unt_loc, unz_loc
        del Tn_device, nn_device
        del un_r_device, un_t_device, un_z_device
        cp._default_memory_pool.free_all_blocks()