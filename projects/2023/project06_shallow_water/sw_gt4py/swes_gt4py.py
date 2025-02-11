#!/usr/bin/env python3
import time
import sys
import math
import numpy as np
import gt4py as gt
from gt4py import gtscript
from scipy.stats import norm
import gt4py_functions as gf

  
class Solver:
    """
    NumPy implementation of a solver class for
    Shallow Water Equations over the surface of a sphere (SWES).

    Notation employed:
    * h			fluid height
    * hs		terrain height (topography)
    * ht		total height (ht = hs + h)
    * phi		longitude
    * R			planet radius
    * theta		latitude
    * u			longitudinal fluid velocity
    * v			latitudinal fluid velocity
    """


    def __init__(self, T, M, N, IC, CFL, diffusion, backend='numpy'):
        """
        Constructor.

        :param	T:	simulation length [days]
        :param	M:	number of grid points along longitude
        :param	N:	number of grid points along latitude
        :param	IC:	tuple storing in first position the ID of the initial condition,
                    then possibly needed parameters. Options for ID:
                    * 0: test case 6 by Williamson
                    * 1: test case 2 by Williamson
        :param	CFL:	CFL number
        :param	diffusion:
                    * TRUE to take viscous diffusion into account,
                    * FALSE otherwise
        """
        
        # --- Build grid --- #

        assert ((M > 1) and (N > 1)), "Number of grid points along each direction must be greater than one."

        # Discretize longitude
        self.M = M
        self.dphi = 2.0 * math.pi / self.M
        self.phi1D = np.linspace(-self.dphi, 2.0*math.pi + self.dphi, self.M+3)

        # Discretize latitude
        # Note: we exclude the poles and only consider a channel from -85 S to 85 N to avoid pole problem
        # Note: the number of grid points must be even to prevent f to vanish
        #       (important for computing initial height and velocity in geostrophic balance)
        if (N % 2 == 0):
            self.N = N
        else:
            self.N = N + 1
            print('Warning: Number of grid points along latitude has been increased by one unit so to make it even.')
        self.theta_range = 85.0
        self.dtheta = (2*self.theta_range/180.0) * math.pi / (self.N - 1)
        self.theta1D = np.linspace(-self.theta_range/180.0*math.pi, self.theta_range/180.0*math.pi, self.N)

        # Build grid
        self.phi, self.theta = np.meshgrid(self.phi1D, self.theta1D, indexing = 'ij')

        # Cosine of mid-point values for theta along y
        self.c = np.cos(self.theta)
        #self.cMidy = np.cos(0.5 * (self.theta[1:-1,1:] + self.theta[1:-1,:-1]))
        
        self.cMidy = np.cos(0.5 * (self.theta[:,1:] + self.theta[:,:-1]))

        self.tg = np.tan(self.theta[1:-1,1:-1])
        self.tgMidx = np.tan(0.5 * (self.theta[:-1,:] + self.theta[1:,:]))
        self.tgMidy = np.tan(0.5 * (self.theta[:,:-1] + self.theta[:,1:]))

        # --- Set planet's constants --- #

        self.setPlanetConstants()

        # --- Cartesian coordinates and increments --- #

        # Coordinates
        self.x	= self.a * np.cos(self.theta) * self.phi
        self.y	= self.a * self.theta
        self.y1 = self.a * np.sin(self.theta)

        # Increments
        self.dx  = self.x[1:,:] - self.x[:-1,:] # x(1)-x(0)
        self.dy  = self.y[:,1:] - self.y[:,:-1]
        self.dy1 = self.y1[:,1:] - self.y1[:,:-1]

        # Compute mimimum distance between grid points on the sphere.
        # This will be useful for CFL condition
        self.dxmin = self.dx.min()
        self.dymin = self.dy.min()

        # "Centred" increments. Useful for updating solution
        # with Lax-Wendroff scheme
        self.dxc  = 0.5 * (self.dx[:-1,1:-1] + self.dx[1:,1:-1])
        self.dyc  = 0.5 * (self.dy[1:-1,:-1] + self.dy[1:-1,1:])
        self.dy1c = 0.5 * (self.dy1[1:-1,:-1] + self.dy1[1:-1,1:])

        # --- Time discretization --- #

        assert(T >= 0), "Final time must be non-negative."

        # Convert simulation length from days to seconds
        self.T = 24.0 * 3600.0 * T

        # CFL number; this will be used to determine the timestep
        # at each iteration
        self.CFL = CFL

        # --- Terrain height --- #

        # Note: currently just a flat surface
        self.hs = np.zeros((self.M+3, self.N), float)

        # --- Set initial conditions --- #

        assert(IC in range(2)), "Invalid problem ID. See code documentation for implemented initial conditions."
        self.IC = IC
        self.setInitialConditions()

        # --- Setup diffusion --- #
        self.diffusion = diffusion

        # Pre-compute coefficients of second-order approximations of first-order derivative
        if (self.diffusion):
            # Centred finite difference along longitude
            # Ax, Bx and Cx denote the coefficients associated
            # to the centred, upwind and downwind point, respectively
            self.Ax = (self.dx[1:,1:-1] - self.dx[:-1,1:-1]) / (self.dx[1:,1:-1] * self.dx[:-1,1:-1])
            self.Ax = np.concatenate((self.Ax[-2:-1,:], self.Ax, self.Ax[1:2,:]), axis = 0)

            self.Bx = self.dx[:-1,1:-1] / (self.dx[1:,1:-1] * (self.dx[1:,1:-1] + self.dx[:-1,1:-1]))
            self.Bx = np.concatenate((self.Bx[-2:-1,:], self.Bx, self.Bx[1:2,:]), axis = 0)

            self.Cx = - self.dx[1:,1:-1] / (self.dx[:-1,1:-1] * (self.dx[1:,1:-1] + self.dx[:-1,1:-1]))
            self.Cx = np.concatenate((self.Cx[-2:-1,:], self.Cx, self.Cx[1:2,:]), axis = 0)

            # Centred finite difference along latitude
            # Ay, By and Cy denote the coefficients associated
            # to the centred, upwind and downwind point, respectively
            self.Ay = (self.dy[1:-1,1:] - self.dy[1:-1,:-1]) / (self.dy[1:-1,1:] * self.dy[1:-1,:-1])
            self.Ay = np.concatenate((self.Ay[:,0:1], self.Ay, self.Ay[:,-1:]), axis = 1)

            self.By = self.dy[1:-1,:-1] / (self.dy[1:-1,1:] * (self.dy[1:-1,1:] + self.dy[1:-1,:-1]))
            self.By = np.concatenate((self.By[:,0:1], self.By, self.By[:,-1:]), axis = 1)

            self.Cy = - self.dy[1:-1,1:] / (self.dy[1:-1,:-1] * (self.dy[1:-1,1:] + self.dy[1:-1,:-1]))
            self.Cy = np.concatenate((self.Cy[:,0:1], self.Cy, self.Cy[:,-1:]), axis = 1)

        # --- GT4Py settings --- #
        
        # self.num_halo = 1 # not needed
        self.backend = backend # default: 'numpy'
        
        nx, ny = np.shape(self.h)
        nz = 1
        
        self.shape             = (  nx,   ny, nz) # full domain size w/ halo
        self.default_shape     = (nx-2, ny-2, nz) # physical domain size w/o halo
        self.extended_shape    = (nx+2, ny+2, nz) # full domain size w/ two halos
        self.shape_staggered_x = (nx-1,   ny, nz) # full but staggered in x
        self.shape_staggered_y = (  nx, ny-1, nz) # full but staggered in y

        self.default_origin  =  (0,0,0)
        self.origin_staggered = (0,0,0)
        
        # --- move all fields to GT4Py --- #
        
        # coordinates
        self.theta =  gt.storage.from_array(np.expand_dims(self.theta,axis=2), self.backend, self.default_origin)
        self.phi =    gt.storage.from_array(np.expand_dims(self.phi,axis=2), self.backend, self.default_origin)
        self.c =      gt.storage.from_array(np.expand_dims(self.c,axis=2), self.backend, self.default_origin)
        self.cMidy =  gt.storage.from_array(np.expand_dims(self.cMidy,axis=2), self.backend, self.default_origin)
        self.tg =     gt.storage.from_array(np.expand_dims(self.tg,axis=2), self.backend, self.default_origin)
        self.tgMidx = gt.storage.from_array(np.expand_dims(self.tgMidx,axis=2), self.backend, self.default_origin)
        self.tgMidy = gt.storage.from_array(np.expand_dims(self.tgMidy,axis=2), self.backend, self.default_origin)
        self.f =      gt.storage.from_array(np.expand_dims(self.f,axis=2), self.backend, self.default_origin)
        
        # grid spacing
        self.dx =    gt.storage.from_array(np.expand_dims(self.dx,axis=2), self.backend, self.default_origin)
        self.dy =    gt.storage.from_array(np.expand_dims(self.dy,axis=2), self.backend, self.default_origin)
        self.dy1 =   gt.storage.from_array(np.expand_dims(self.dy1,axis=2), self.backend, self.default_origin)
        self.dxc =   gt.storage.from_array(np.expand_dims(self.dxc,axis=2), self.backend, self.default_origin)
        self.dyc =   gt.storage.from_array(np.expand_dims(self.dyc,axis=2), self.backend, self.default_origin)
        self.dy1c =  gt.storage.from_array(np.expand_dims(self.dy1c,axis=2), self.backend, self.default_origin)
        
        # terrain height (zero)
        self.hs =    gt.storage.from_array(np.expand_dims(self.hs,axis=2), self.backend, self.default_origin)
        
        # initial condition
        self.h =     gt.storage.from_array(np.expand_dims(self.h,axis=2), self.backend, self.default_origin)
        self.u =     gt.storage.from_array(np.expand_dims(self.u,axis=2), self.backend, self.default_origin)
        self.v =     gt.storage.from_array(np.expand_dims(self.v,axis=2), self.backend, self.default_origin)
        
        # empty fields
        self.hu =    gt.storage.empty(self.backend, self.default_origin, self.shape, dtype=float)
        self.hv =    gt.storage.empty(self.backend, self.default_origin, self.shape, dtype=float)
        self.hv1 =   gt.storage.empty(self.backend, self.default_origin, self.shape, dtype=float)
        self.v1 =    gt.storage.empty(self.backend, self.default_origin, self.shape, dtype=float)
        
        # temporary staggered fields
        self.hMidx = gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_x, dtype=float)
        self.huMidx= gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_x, dtype=float)
        self.hvMidx= gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_x, dtype=float)
        
        self.hMidy=  gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_y, dtype=float)
        self.huMidy= gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_y, dtype=float)
        self.hvMidy= gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_y, dtype=float)
        
        self.VxMidnew =  gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_x, dtype=float)
        self.Vy1Midnew = gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_y, dtype=float)
        self.Vy2Midnew = gt.storage.empty(self.backend, self.default_origin, self.shape_staggered_y, dtype=float)
        
        if self.diffusion:
            # print('Converting Diffusion')
            # diffusion
            self.Ax = gt.storage.from_array(np.expand_dims(self.Ax,axis=2), self.backend, self.default_origin)
            self.Bx = gt.storage.from_array(np.expand_dims(self.Bx,axis=2), self.backend, self.default_origin)
            self.Cx = gt.storage.from_array(np.expand_dims(self.Cx,axis=2), self.backend, self.default_origin)
            self.Ay = gt.storage.from_array(np.expand_dims(self.Ay,axis=2), self.backend, self.default_origin)
            self.By = gt.storage.from_array(np.expand_dims(self.By,axis=2), self.backend, self.default_origin)
            self.Cy = gt.storage.from_array(np.expand_dims(self.Cy,axis=2), self.backend, self.default_origin)
            self.qlap =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float) 
            
            self.hext =  gt.storage.empty(self.backend, self.default_origin, self.extended_shape, dtype=float) 
            self.uext =  gt.storage.empty(self.backend, self.default_origin, self.extended_shape, dtype=float) 
            self.vext =  gt.storage.empty(self.backend, self.default_origin, self.extended_shape, dtype=float) 
            
            
        # output fields
        self.htemp=  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)        
        self.utemp =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)
        self.vtemp =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)
        
        self.hnew =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)        
        self.unew =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)
        self.vnew =  gt.storage.empty(self.backend, self.default_origin, self.default_shape, dtype=float)
        
        # --- compile  stencils --- #
        
        kwargs = {"verbose": True} if self.backend in ("gtx86", "gtmc", "gtcuda") else {}
        
        self.temp_variables = gtscript.stencil(
            definition=gf.compute_temp_variables,
            backend=self.backend,
            rebuild=False,
            **kwargs
        )
        
        self.x_staggered = gtscript.stencil(
            definition=gf.x_staggered_first_step, 
            backend=self.backend, 
            externals={
                "compute_hMidx":gf.compute_hMidx, 
                "compute_huMidx":gf.compute_huMidx,
                "compute_hvMidx":gf.compute_hvMidx
            }, 
            rebuild=False,
            **kwargs
        )
        
        self.y_staggered = gtscript.stencil(
            definition=gf.y_staggered_first_step, 
            backend=self.backend, 
            externals={
                "compute_hMidy":gf.compute_hMidy, 
                "compute_huMidy":gf.compute_huMidy,
                "compute_hvMidy":gf.compute_hvMidy
            },
            rebuild=False,
            **kwargs
        )
        
        self.combined = gtscript.stencil(
            definition=gf.combined_last_step,
            backend=self.backend, 
            externals={
                "compute_hnew":gf.compute_hnew,
                "compute_hunew":gf.compute_hunew, 
                "compute_hvnew":gf.compute_hvnew
            },
            rebuild=False,
            **kwargs
        )
        
        if self.diffusion:
            self.laplacian = gtscript.stencil(definition=gf.compute_Lapacian, backend=self.backend, rebuild=False,  **kwargs)

    def setPlanetConstants(self):
        """
        Set Earth's constants.

        :attribute	g:				gravity	[m/s2]
        :attribute	rho:			average atmosphere density	[kg/m3]
        :attribute	a:				average radius	[m]
        :attribute	omega:			rotation rate	[Hz]
        :attribute	scaleHeight:	atmosphere scale height	[m]
        :attribute	nu:				viscosity	[m2/s]
        :attribute	f:				Coriolis parameter	[Hz]

        :param:

        :return:
        """

        # Earth
        self.g				= 9.80616
        self.rho			= 1.2
        self.a				= 6.37122e6
        self.omega			= 7.292e-5
        self.scaleHeight	= 8.0e3
        self.nu				= 5.0e5

        # Coriolis parameter
        self.f = 2.0 * self.omega * np.sin(self.theta)


    def setInitialConditions(self):
        """
        Set initial conditions.

        :attribute	h:	initial fluid height
        :attribute	u:	initial longitudinal velocity
        :attribute	v:	initial latitudinal velocity

        :param:

        :return:
        """

        # --- IC 0: sixth test case taken from Williamson's suite --- #
        # ---       Rossby-Haurwitz Wave                          --- #

        if (self.IC == 0):
            # Set constants
            w  = 7.848e-6
            K  = 7.848e-6
            h0 = 8e3
            R  = 4.0

            # Compute initial fluid height
            A = 0.5 * w * (2.0 * self.omega + w) * (np.cos(self.theta) ** 2.0) + \
                0.25 * (K ** 2.0) * (np.cos(self.theta) ** (2.0 * R)) * \
                ((R + 1.0) * (np.cos(self.theta) ** 2.0) + \
                 (2.0 * (R ** 2.0) - R - 2.0) - \
                 2.0 * (R ** 2.0) * (np.cos(self.theta) ** (-2.0)))
            B = (2.0 * (self.omega + w) * K) / ((R + 1.0) * (R + 2.0)) * \
                (np.cos(self.theta) ** R) * \
                (((R ** 2.0) + 2.0 * R + 2.0) - \
                 ((R + 1.0) ** 2.0) * (np.cos(self.theta) ** 2.0))
            C = 0.25 * (K ** 2.0) * (np.cos(self.theta) ** (2.0 * R)) * \
                ((R + 1.0) * (np.cos(self.theta) ** 2.0) - (R + 2.0))

            h = h0 + ((self.a ** 2.0) * A + \
                      (self.a ** 2.0) * B * np.cos(R * self.phi) + \
                      (self.a ** 2.0) * C * np.cos(2.0 * R * self.phi)) / self.g

            # Compute initial wind
            u = self.a * w * np.cos(self.theta) + \
                self.a * K * (np.cos(self.theta) ** (R - 1.0)) * \
                (R * (np.sin(self.theta) ** 2.0) - (np.cos(self.theta) ** 2.0)) * \
                np.cos(R * self.phi)
            v = - self.a * K * R * (np.cos(self.theta) ** (R - 1.0)) * \
                  np.sin(self.theta) * np.sin(R * self.phi)

        # --- IC 1: second test case taken from Williamson's suite --- #
        # ----      Steady State Nonlinear Zonal Geostrophic Flow  --- #

        elif (self.IC == 1):
            # Suggested values for $\alpha$ for second
            # test cases of Williamson's suite:
            #	- 0
            #	- 0.05
            #	- pi/2 - 0.05
            #	- pi/2
            alpha = math.pi/2

            # Set constants
            u0 = 2.0 * math.pi * self.a / (12.0 * 24.0 * 3600.0)
            h0 = 2.94e4 / self.g

            # Make Coriolis parameter dependent on longitude and latitude
            self.f = 2.0 * self.omega * \
                     (- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) + \
                      np.sin(self.theta) * np.cos(alpha))

            # Compute initial height
            h = h0 - (self.a * self.omega * u0 + 0.5 * (u0 ** 2.0)) * \
                     ((- np.cos(self.phi) * np.cos(self.theta) * np.sin(alpha) + \
                       np.sin(self.theta) * np.cos(alpha)) ** 2.0) / self.g

            # Compute initial wind
            u = u0 * (np.cos(self.theta) * np.cos(alpha) + \
                      np.cos(self.phi) * np.sin(self.theta) * np.sin(alpha))
            self.uMidx = u0 * (np.cos(0.5 * (self.theta[:-1,:] + self.theta[1:,:])) * np.cos(alpha) + \
                                 np.cos(0.5 * (self.phi[:-1,:] + self.phi[1:,:])) * \
                                 np.sin(0.5 * (self.theta[:-1,:] + self.theta[1:,:])) * np.sin(alpha))
            self.uMidy = u0 * (np.cos(0.5 * (self.theta[:,:-1] + self.theta[:,1:])) * np.cos(alpha) + \
                                 np.cos(0.5 * (self.phi[:,:-1] + self.phi[:,1:])) * \
                                 np.sin(0.5 * (self.theta[:,:-1] + self.theta[:,1:])) * np.sin(alpha))

            v = - u0 * np.sin(self.phi) * np.sin(alpha)
            self.vMidx = - u0 * np.sin(0.5 * (self.phi[:-1,:] + self.phi[1:,:])) * np.sin(alpha)
            self.vMidy = - u0 * np.sin(0.5 * (self.phi[:,:-1] + self.phi[:,1:])) * np.sin(alpha)

        self.h = h
        self.u = u
        self.v = v
    
    
    def solve(self, verbose, save):
        """
        Solver.

        :param	verbose:	if positive, print to screen information about the solution
                            every 'verbose' timesteps
        :param	save:	if positive, store the solution every 'save' timesteps

        :return	h:	if save <= 0, fluid height at final time
        :return	u:	if save <= 0, fluid longitudinal velocity at final time
        :return	v:	if save <= 0, fluid latitudinal velocity at final time
        :return tsave:	if save > 0, stored timesteps
        :return	phi:	if save > 0, longitudinal coordinates of grid points
        :return theta:	if save > 0, latitudinal coordinates of grid points
        :return	hsave:	if save > 0, stored fluid height
        :return	usave:	if save > 0, stored longitudinal velocity
        :return	vsave:	if save > 0, stored latitudinal velocity
        """

        verbose = int(verbose)
        save = int(save)

        # --- Print and save --- #

        # Print to screen
        if (verbose > 0):
            norm = np.sqrt(self.u*self.u + self.v*self.v)
            umax = norm.max()
            print("Time = %6.2f hours (max %i); max(|u|) = %8.8f" \
                    % (0.0, int(self.T / 3600.0), umax))

        # Save
        if (save > 0):
            tsave = np.array([[0.0]])
            hsave = self.h[1:-1, :, :]
            usave = self.u[1:-1, :, :]
            vsave = self.v[1:-1, :, :]

        # --- Time marching --- #

        n = 0
        t = 0.0
        wall_zero = time.time()
        
        while (t < self.T):
            
            # Update number of iterations
            n += 1

            # --- Compute timestep through CFL condition --- #

            # Compute flux Jacobian eigenvalues
            eigenx = (np.maximum(np.absolute(self.u - np.sqrt(self.g * np.absolute(self.h))),
                                 np.maximum(np.absolute(self.u),
                                             np.absolute(self.u + np.sqrt(self.g * np.absolute(self.h)))))).max()

            eigeny = (np.maximum(np.absolute(self.v - np.sqrt(self.g * np.absolute(self.h))),
                                 np.maximum(np.absolute(self.v),
                                             np.absolute(self.v + np.sqrt(self.g * np.absolute(self.h)))))).max()

            # Compute timestep
            dtmax = np.minimum(self.dxmin/eigenx, self.dymin/eigeny)
            self.dt = self.CFL * dtmax
            
            #Convert to numpy
            self.dt=float(np.asarray(self.dt))
            
            
            # If needed, adjust timestep not to exceed final time
            if (t + self.dt > self.T):
                self.dt = self.T - t
                t = self.T
            else:
                t += self.dt
            
            # --- Update solution --- #
            
            self.temp_variables(
                u=self.u,v=self.v,h=self.h,
                c=self.c,
                hu=self.hu,hv=self.hv,v1=self.v1,
                origin=self.default_origin, domain=self.shape
            )
            
            self.x_staggered(
                u=self.u,v=self.v,h=self.h,
                hu=self.hu,hv=self.hv,
                f=self.f,
                dx=self.dx,
                tgMidx=self.tgMidx,hMidx=self.hMidx,huMidx=self.huMidx,hvMidx=self.hvMidx,
                dt=self.dt,
                g=self.g,a=self.a,
                origin=self.default_origin, domain=self.shape_staggered_x
            )

            self.y_staggered(
                u=self.u,v=self.v,h=self.h,
                hu=self.hu,hv=self.hv,
                v1=self.v1,
                f=self.f,
                dy=self.dy,dy1=self.dy1,
                tgMidy=self.tgMidy,hMidy=self.hMidy,huMidy=self.huMidy,hvMidy=self.hvMidy,
                dt=self.dt,
                g=self.g,a=self.a, 
                origin=self.default_origin, domain=self.shape_staggered_y
            )
            
            
            self.combined(
                h=self.h,hu=self.hu,hv=self.hv,
                hs=self.hs, 
                f=self.f,
                tg=self.tg,
                huMidx=self.huMidx,huMidy=self.huMidy,
                hvMidx=self.hvMidx,hvMidy=self.hvMidy,
                hMidx=self.hMidx,hMidy=self.hMidy,
                cMidy=self.cMidy, 
                dx=self.dx, dy1=self.dy1,dxc=self.dxc,dyc=self.dyc,dy1c=self.dy1c,
                hnew=self.htemp, unew=self.utemp, vnew=self.vtemp, 
                VxMidnew=self.VxMidnew, Vy1Midnew=self.Vy1Midnew, Vy2Midnew=self.Vy2Midnew,
                dt=self.dt,
                g=self.g, a=self.a,
                origin=self.default_origin, domain=self.default_shape
            )
            

            if (self.diffusion):
                # Extend fluid height
                self.hext = np.concatenate((self.h[-4:-3,:], self.h, self.h[3:4,:]), axis = 0)
                self.hext = np.concatenate((self.hext[:,0:1,:], self.hext, self.hext[:,-1:,:]), axis = 1)
                
                self.hext = gt.storage.from_array(self.hext, self.backend, self.default_origin)
                
                # Compute Laplacian
                self.laplacian(
                    q=self.hext, 
                    Ax=self.Ax, Bx=self.Bx, Cx=self.Cx, 
                    Ay=self.Ay, By=self.By, Cy=self.Cy, 
                    qtemp=self.htemp, qnew=self.hnew, dt=self.dt, nu=self.nu,
                    origin=self.default_origin,
                    domain=self.default_shape
                )

                # Add the Laplacian
                #self.hnew = self.hnew[:,:,:] + self.dt * self.nu * self.qlap[:,:,:]
                
                # Extend longitudinal velocity
                self.uext = np.concatenate((self.u[-4:-3,:], self.u, self.u[3:4,:]), axis = 0)
                self.uext = np.concatenate((self.uext[:,0:1,:], self.uext, self.uext[:,-1:,:]), axis = 1)
                
                self.uext = gt.storage.from_array(self.uext, self.backend, self.default_origin)
            
                # Compute Laplacian
                self.laplacian(
                    q=self.uext, 
                    Ax=self.Ax, Bx=self.Bx, Cx=self.Cx, 
                    Ay=self.Ay, By=self.By, Cy=self.Cy, 
                    qtemp=self.utemp, qnew=self.unew, dt=self.dt, nu=self.nu,
                    origin=self.default_origin,
                    domain=self.default_shape
                )
                # Add the Laplacian
                #self.unew = self.unew[:,:,:] + self.dt * self.nu * self.qlap[:,:,:]

                # Extend fluid height
                self.vext = np.concatenate((self.v[-4:-3,:], self.v, self.v[3:4,:]), axis = 0)
                self.vext = np.concatenate((self.vext[:,0:1,:], self.vext, self.vext[:,-1:,:]), axis = 1)
                
                self.vext = gt.storage.from_array(self.vext, self.backend, self.default_origin)

                # Compute Laplacian
                self.laplacian(
                    q=self.vext, 
                    Ax=self.Ax, Bx=self.Bx, Cx=self.Cx, 
                    Ay=self.Ay, By=self.By, Cy=self.Cy, 
                    qtemp=self.vtemp, qnew=self.vnew, dt=self.dt, nu=self.nu,
                    origin=self.default_origin, 
                    domain=self.default_shape
                )
                # Add the Laplacian
                #self.vnew = self.vnew[:,:,:]
            
            else:
                self.hnew=self.htemp
                self.unew=self.utemp
                self.vnew=self.vtemp
            
            # --- Update solution applying BCs --- #
            self.h[:,1:-1] = np.concatenate((self.hnew[-2:-1,:], self.hnew, self.hnew[1:2,:]), axis = 0)
            self.h[:,0]  = self.h[:,1]
            self.h[:,-1] = self.h[:,-2]
            self.u[:,1:-1] = np.concatenate((self.unew[-2:-1,:], self.unew, self.unew[1:2,:]), axis = 0)
            self.u[:,0]  = self.u[:,1]
            self.u[:,-1] = self.u[:,-2]

            self.v[:,1:-1] = np.concatenate((self.vnew[-2:-1,:], self.vnew, self.vnew[1:2,:]), axis = 0)
            self.v[:,0]  = self.v[:,1]
            self.v[:,-1] = self.v[:,-2]

            # --- Print and save --- #

            if (verbose > 0 and (n % verbose == 0)):
                norm = np.sqrt(self.u*self.u + self.v*self.v)
                umax = norm.max()
                wall_time = time.time() - wall_zero
                print(f"\nIteration {n=}; {wall_time=:2.2f}s, saving = {n%save==0}")
                print(f"Time = {t/3600:6.2f} hours (max {int(self.T / 3600.0)}); max(|u|) = {umax:6.4f}")

            if (save > 0 and (n % save == 0)):
                tsave = np.concatenate((tsave, np.array([[t]])), axis = 0)
                
                hsave = np.concatenate((hsave, self.h[1:-1, :, :]), axis = 2)
                usave = np.concatenate((usave, self.u[1:-1, :, :]), axis = 2)
                vsave = np.concatenate((vsave, self.v[1:-1, :, :]), axis = 2)
        
        wall_time = time.time() - wall_zero
        
        # --- Back to numpy format --- #
        
        self.phi = np.asarray(self.phi)
        self.theta = np.asarray(self.theta)
        self.h = np.asarray(self.h)
        self.u = np.asarray(self.u)
        self.v = np.asarray(self.v)
        hsave = np.asarray(hsave)
        usave = np.asarray(usave)
        vsave = np.asarray(vsave)
        
        # --- Return --- #

        if (save > 0):
            return wall_time, tsave, self.phi[:,:,0], self.theta[:,:,0], hsave, usave, vsave
        else:
            return wall_time, self.h[:,:,0], self.u[:,:,0], self.v[:,:,0]
