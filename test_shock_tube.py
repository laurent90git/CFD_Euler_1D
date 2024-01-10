#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:45:27 2020

This script test the 1D Euler solver with shock tube configuration.
The two side of the tube are originally at rest, but with different pressures
and densities. The removal of the membrane separating both sides gives rise
to a system of shock and rarefaction waves, whose solution can be computed
analytically, at least up to the moment where the waves are reflected on the
walls.

@author: laurent
"""
from Euler_FV_scheme import modelfun, setupFiniteVolumeMesh, Riemann_exact, plotfuncustom,computeT, computeP, getXFromVars, getVarsFromX, getVarsFromX_vectorized, computeOtherVariables
from Euler_FV_scheme import cv, r, gamma
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import scipy.optimize

options={'mesh':{},
         'BCs':{'left':{'type':''},
                'right':{'type':''}},
         'scheme':{}}

# Choose the Riemann solver and the spatial reconsturction
options['scheme']['riemann_solver'] = 3
options['scheme']['limiter'] = 1
options['scheme']['reconstruction'] = 1

# Specify boundary conditions
options['BCs']['left']['type']  = "reflective"
options['BCs']['right']['type'] = "reflective"

# Generate the faces of the mesh
xfaces = np.linspace(-1,1,100)

# Generate the corresponding mesh
options['mesh'] = setupFiniteVolumeMesh(xfaces,{})
xcells = options['mesh']['cellX']
nx = xcells.size
options['mesh']['nx']=nx

# Generate the initial condition
xc = 0.
P_0 = np.zeros_like(xcells)
P_0[xcells<xc]  =  1.*1e5
P_0[xcells>=xc] =  0.1*1e5

rho_0 = np.zeros_like(xcells)
rho_0[xcells<xc]  =  1.0
rho_0[xcells>=xc] =  0.125

u_0 = np.zeros_like(xcells)
# u_0[xcells<xc]  =  100.
# u_0[xcells>=xc] =  50.

T_0 = computeT(P_0, rho_0)
E_0 = cv*T_0 + 0.5*u_0*u_0

X0 = getXFromVars(rho_0, rho_0*u_0, rho_0*E_0)

rho,rhoU,rhoE = getVarsFromX(x=X0, options=options)
assert np.all(rho==rho_0) # check that the state vector manipulations are coherent

# Specifiy the physical duration of the simulation
tend= 0.01



#%% Create a method for computing the Jacobian in an optimised manner, exploiting its sparsity pattern
import scipy.sparse
import scipy.optimize._numdiff
uband=6; lband=-uband
offsets = [i for i in range(lband,uband)]
sparsity_pattern = scipy.sparse.diags(diagonals=[np.ones((3*nx - abs(i))) for i in offsets], offsets=offsets) 

jacfun = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                    fun=lambda y: modelfun(t=t,x=y,options=options),
                    x0=x, method='2-point', sparsity=sparsity_pattern,
                    rel_step=1e-8, abs_step=1e-8)

if 0: # test correctness of the sparse Jacobian against a naive dense estimation
  jacfun_full = lambda t,x: scipy.optimize._numdiff.approx_derivative(
                      fun=lambda y: modelfun(t=t,x=y,options=options),
                      x0=x, method='2-point', sparsity=None,
                      rel_step=1e-8, abs_step=1e-8)
  
  xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)
  
  jac_full = jacfun_full(0.,xtest)
  jac_sparse = jacfun(0.,xtest)
  assert np.max(np.abs(jac_sparse-jac_full)) < 1e-12, 'The sparse Jacobian estimation is not correct'

#%%
if 0:
    #%% JACOBIAN ANALYSIS (sparsity pattern, eigenvalues)
    tempjac = jacfun
    # tempfun = lambda x: getVarsFromX(x=x, options=options)[0] # rho
    # tempjac = lambda t,x: scipy.optimize._numdiff.approx_derivative(
    #                 fun=tempfun,
    #                 x0=x, method='2-point', sparsity=None,
    #                 rel_step=1e-8)
    Xtest = X0 + np.random.rand(X0.size).reshape(X0.shape)*(1e-3 + 1e-3*X0)

    Jac = np.array(tempjac(0., Xtest))
    # Jac = jacfun(out.t[-1], out.y[:,-1])
    plt.figure()
    plt.spy(Jac)
    n_rank_jac = np.linalg.matrix_rank(Jac),
    plt.title('Jacobian (rank={}, shape={})'.format(n_rank_jac, np.shape(Jac)))
    plt.show()
    if n_rank_jac[0]!=np.size(Jac,1):
        print('The following rows of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=1) ) ))
        print('The following columns of the Jacobian are nil:\n\t{}'.format( np.where( (Jac==0).all(axis=0) ) ))
    if np.size(Jac,1)<500:
        try:
            eigvals, eigvecs= np.linalg.eig(Jac)
            plt.figure()
            plt.scatter(np.real(eigvals), np.imag(eigvals))
            plt.title('Eigenvalues')
        except Exception as e:
            print('caught exception "{}" while computing eigenvalues of the Jacobian'.format(e))
    else:
        print('Skipping eigenvalues computation due to matrix size')
    raise Exception('debug jac')
#%% Analyze the initial time derivatives (debug)
if 0:
  dxdt0 = modelfun(0., X0, options)
  dtrho0, dtrhoU0, dtrhoE0 = (dxdt0[i*nx:(i+1)*nx] for i in range(3))

#%% NUMERICAL INTEGRATION
# jacfun=None
if 0:
  out  = integrate.solve_ivp(fun=lambda t,x: modelfun(t,x,options), t_span=(0.,tend), y0=X0, first_step=1e-9,
                              max_step=np.inf, method='BDF', atol=1e-5, rtol=1e-5, band=(6,6), jac=jacfun,
                              # t_eval=np.arange(0, tend, 5e-5),
                              )
  print('{} time steps'.format(len(out.t)))
else:
  from Euler_FV_scheme import CFLintegration
  out = CFLintegration(fun=lambda t,x: modelfun(t,x,options),t_span=(0.,tend),cfl=0.3,y0=X0,
                 methodclass=None, max_step=np.inf,
                nmax_step=np.inf, relvar_min=None, datalogger=None,
                jacband=None, limitVars=False,events=None, options=options,
                logger=print, log_every=0.)
# #%% backup the result
# from utilities import NumpyEncoder
# import json
# outdict= {"options": options,
#           "y": out.y,
#           't': out.t}
# with open('backup2.json','w') as f: #TODO: binary file to save space
#   json.dump(outdict, f, cls=NumpyEncoder)


#%% GATHER RESULTS
rho, rhoU, rhoE = getVarsFromX_vectorized(out.y, options)
temp = computeOtherVariables(rho, rhoU, rhoE)
u,T,P = temp['u'],temp['T'],temp['P']
time = out.t
# u[i,:] corresponds to the velocity field at the i-th time step

#%% Plot the solution at multiple instants
# selected_time_indices = [-1] #[ i.astype(int) for i in np.linspace(0,time.size-1,20) ]
selected_time_indices = range(time.size)
plotfuncustom(xcells, P.T, time, 'P', selected_time_indices, marker=None)
plotfuncustom(xcells, u.T, time, 'u', selected_time_indices, marker=None)
plotfuncustom(xcells, rho.T, time, 'rho', selected_time_indices, marker=None)
plotfuncustom(xcells, rhoU.T, time, 'rhoU', selected_time_indices, marker=None)
plotfuncustom(xcells, rhoE.T, time, 'rhoE', selected_time_indices, marker=None)
plotfuncustom(xcells, T.T, time, 't', selected_time_indices, marker=None)


#%% Compare with analytical solution
# /!\ The analytical solution assumes the initial discontinuity is at x=0
mesh=options['mesh']['cellX']
mesh_exact = np.linspace(np.min(mesh), np.max(mesh), int(2e3))
exactsol = Riemann_exact(t=time[-1], g=gamma,
                         Wl=np.array([rho_0[0], u_0[0], P_0[0]]),
                         Wr=np.array([rho_0[-1], u_0[-1], P_0[-1]]),
                         grid=mesh_exact)
rho_exact = exactsol[0]
u_exact = exactsol[1]
P_exact = exactsol[2]
T_exact = P_exact/rho_exact/r


plt.figure()
plt.plot(mesh_exact, rho_exact, color='r', label='exact')
plt.plot(mesh, rho[:,-1], color='b', label='simulation', marker='+', linestyle='')
plt.xlabel('x (m)')
plt.ylabel(r'$\rho$ (kg.m$^{-3}$)')
plt.title('Density')
plt.legend()
plt.xlim(-1,1); plt.ylim(0,1.1); plt.grid()
# plt.savefig('comparison_density_FVS_r1l1.png', dpi=300)

#%%
plt.figure()
plt.plot(mesh_exact, u_exact, color='r', label='exact')
plt.plot(mesh, u[:,-1], color='b', label='num', marker='+', linestyle='')
plt.xlabel('x (m)')
plt.ylabel(r'$u$')
plt.title('Velocity (m/s)')

plt.figure()
plt.plot(mesh_exact, P_exact, color='r', label='exact')
plt.plot(mesh, P[:,-1], color='b', label='num', marker='+', linestyle='')
plt.xlabel('x (m)')
plt.ylabel('P (Pa)')
plt.title('Pressure')

#%% Check that the integral of the total energy is conserved
total_E = np.sum(rhoE.T * options['mesh']['cellSize'], axis=1)
plt.figure(dpi=300)
plt.plot(time, (total_E-total_E[0])/total_E[0])
plt.grid()
plt.xlabel('t (s)')
plt.ylabel(r'Relative energy loss')
plt.title('Total energy decay')
plt.savefig('total_E_decay.png')

#%% COmpute Schlieren field
xx,yy = np.meshgrid(mesh,time)
# compute nmerical schlieren
schlieren =np.gradient((P/rho).T, axis=1)/np.gradient(xx, axis=1)

#%%
# histdata = plt.hist( np.abs(schlieren).flatten(), bins=np.linspace(0, np.max(np.abs(schlieren)), 100) )
# proportions = histdata[0]/sum(histdata[0])
# xbins = histdata[1]

# plt.figure()
# plt.semilogy(xbins[:-1], proportions)

# plt.figure()
# plt.plot(xbins[:-1], 1-np.cumsum(proportions))
# for val in [0,0.25,0.5,0.75,1.]:
#   plt.axhline(val, color='r')
#%% VISUALIZE (t,X) evolution
time_gap = 10
time_max = np.argmin(np.abs(time-0.2))
space_gap = 1

varplot = (
            (np.log10(np.abs(schlieren)), 'schlieren', 'Synthetic Schlieren', None, None, 'Greys'),
            #(np.sign(schlieren), 'sign_schlieren', 'Sign of synthetic Schlieren', None, None, 'Greys'),
            (schlieren, 'schlieren2', 'Synthetic Schlieren', None, None, 'seismic'),
            (np.abs((schlieren)), 'schlieren_abs', 'Synthetic Schlieren', None, None, 'Greys'),
            (P.T, 'P', 'Pressure field', 'P (Pa)', None, None),
            (u.T, 'u', 'Velocity field', r'$u$ (m/s)', np.max(np.abs(u))*np.array([-1,1]), 'seismic'),
            (np.sign(u).T, 'sign_u', 'Velocity direction', r'sign($u$)', None, None),
            (np.abs(temp['M']).T, 'Mabs', 'Absolute Mach field', r'$|M|$', np.max(np.abs(temp['M']))*np.array([0,1]), None),
            (T.T, 'T', 'Temperature field', 'T (K)', None, 'hot'),
            (rhoE.T, 'rhoE', 'Total energy field', r'$\rho E$ (J.m$^{-3}$)', (np.min(rhoE), 1.01*np.max(rhoE)), None),
           )
nlevels = 990
for var, name, title, clabel, zlims, cmap in varplot:
  plt.figure(dpi=500)
  if zlims is not None:
    levels = np.linspace(zlims[0], zlims[1], nlevels)
  else:
    levels = nlevels
  plt.contourf(xx[:time_max:time_gap,::space_gap],yy[:time_max:time_gap,::space_gap],var[:time_max:time_gap,::space_gap],
                levels=levels, cmap=cmap)
  plt.xlabel(r'$x$ (m)')
  plt.ylabel(r'$t$ (s)')
  if clabel:
    cb = plt.colorbar()
    cb.set_label(clabel)
  # plt.grid()
  plt.title(title)
  plt.savefig('{}_mode2r1l1.png'.format(name), dpi=500)
  plt.show()
  


#%% Visualize exact solution
t_exact = np.linspace(0., time[-1]-time[0],100)+time[0]
exactsol = []
for t in t_exact:
  exactsol.append( Riemann_exact(t=t, g=gamma,
                           Wl=np.array([rho[0,0], u[0,0], P[0,0]]),
                           Wr=np.array([rho[-1,0], u[-1,0], P[-1,0]]),
                           grid=mesh_exact) )
  
#%%
exactsol_array = np.array(exactsol)
rho_exact = exactsol_array[:,0,:]
u_exact = exactsol_array[:,1,:]
P_exact = exactsol_array[:,2,:]
T_exact = P_exact/rho_exact/r

# temp = computeOtherVariables(rho_exact, rho_exact*u_exact, rho_exact*computeE(T_exact,u_exact))
# u_exact,T_exact,P_exact = temp['u'],temp['T'],temp['P']

plt.figure()
xx_exa,yy_exa = np.meshgrid(mesh_exact,t_exact)
# plt.contourf(xx_exa,yy_exa,P_exact, levels=100)
plt.contourf(xx_exa,yy_exa,P_exact, levels=100)
plt.xlabel('x (m)')
plt.ylabel('t (s)')
cb = plt.colorbar()
cb.set_label('P (Pa)')
plt.grid()
plt.title('Exact pressure field')

#%% Test adaptation criterion
options['mesh']['adapt']={"mode":1,
                          "atol":1e-2, "rtol":1e-2}

dx_opt = modelfun(t=out.t[-1], x=out.y[:,-1], options=options, estimate_space_error=True)

plt.figure()
plt.semilogy(options['mesh']['faceX'], dx_opt, label='optimal mesh')
plt.semilogy(options['mesh']['faceX'],
         np.hstack((options['mesh']['cellSize'][0], options['mesh']['cellSize'])),
         label='mesh')
plt.grid()
plt.xlim(options['mesh']['faceX'][0], options['mesh']['faceX'][-1])
plt.legend()