# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 14:56:13 2023

@author: lfrancoi
"""

from Euler_FV_scheme import rk_nssp_53, getCFLtimestep, limitValues, limit_stages, setupFiniteVolumeMesh, computeOtherVariables
import numpy as np
import matplotlib.pyplot as plt
from tree_mesh import Node
from tqdm import tqdm

def space_adapt_integration(fun,t_span, cfl, init_fun, x_interval,
                            options, level0, max_level=15,
                             nmax_step=np.inf, refine_every=1,
                             limitVars=False,events=None,
                             logger=print, log_every=0.,
                             debug=False):
    """ Space adaptive CFL-driven time integration
        Inputs:
            - fun : callable
                model function that returns the time derivative of the state vector y
            - cflfun : callable
                function which returns the time step such that CFL<=1 everywhere
            - cfl : float
                desired CFL number
            - t_span : list, tuple, array_like
                Start and end times
            - y0 : Numpy array
                initial solution
            - method : solver object, e.g. from scipy.integrate import RK23
                only explicit methods should be used here, because of the way the CFL
                time step is enforced
                TODO: other solvers
        Output:
            - sol: bunch object containing the solution history
    """
    from scipy.integrate._ivp.ivp import OdeResult
    
    node1 = Node(center=(x_interval[1]+x_interval[0]),
                 dx=(x_interval[1]-x_interval[0]),
                 value=None,
                 level=0, min_level=level0,
                 max_level=max_level)
    node1.recursive_refine(level0)      # initial mesh
    node1.recursive_set_value(init_fun) # initialise field
    
    # get initial mesh
    faces_old, centers_old, levels_old, y_old, nodes_old = node1.getMesh()
    y_old_reshaped = y_old.T.reshape((-1,), order='F')

    # store mesh structure
    options['mesh'] = setupFiniteVolumeMesh(faces_old,options['mesh'])
    

    t_old = t_span[0]
    
    # initialise storage
    centers_hist=[centers_old]
    faces_hist=[options['mesh']['faceX'].copy()]
    yhist=[y_old]
    thist=[t_old]
    levels_hist = [levels_old]


    if limitVars:
      limitfun = lambda y: limit_stages(y=y, options=options)
    else:
      limitfun = lambda y: y


    last_out_t = t_old
    nfev = 0
    nt=0
    
    try:
      while t_old<t_span[-1]:
  
        # refine mesh
        if np.mod(nt, refine_every)==0:
            bRefineFurther = True
            if nt==0:
              itmax = 10
            else:
              itmax = 1
            for itr in range(itmax):
              if not bRefineFurther:
                break
  
              # TODO: loop ?
              # Compute adaptation criterion  
              dx_opt = fun(t=t_old, x=y_old_reshaped, options=options, estimate_space_error=True)
              # convert to target cell center sizes
              dx_opt_centers = np.minimum(dx_opt[:-1], dx_opt[1:])
              dx_opt_centers = np.minimum( dx_opt_centers, node1.dx)
    
              
              # compute required levels
              dx_centers_old = options['mesh']['cellSize']
              levels_to_add = np.ceil( np.log(dx_centers_old / dx_opt_centers) / np.log(2) ).astype(int)
              assert len(levels_to_add)==len(nodes_old)
              levels_to_add = np.maximum(levels_to_add, -1)
              levels_to_add = np.minimum(levels_to_add, 2)
              
              logger('  levels_to_add = {} to {}'.format(np.min(levels_to_add), np.max(levels_to_add)))
              
              # refine nodes
              for i, (node, lvladd) in enumerate(zip(nodes_old, levels_to_add)):
                # print(i)
                # refresh node values
                # if i>=y_old.shape[0]:
                #   import pdb; pdb.set_trace()
                node.value = y_old[i,:]
                if lvladd<0:
                  node.destroy()
                elif lvladd>0:
                  node.recursive_refine(target_level=node.level + lvladd)
              
              for node in nodes_old:
                if node.is_destroyed:
                  del node
                  
              # grade tree
              node1.gradeTree()
              # TODO: in the previous section, only delete nodes which are not needed for grading...
              node1.recursive_destruction_reset()
      
              
              # TODO: nicer conservative interpolation
              if nt==0:
                node1.recursive_set_value(init_fun) # initialise field
              else:
                # TODO: nice conservative linear interpolation ?
                # bRefineFurther = False # TODO: improve
                pass
              
              # get new mesh data
              old_old_faces   = options['mesh']['faceX']
              old_old_centers = options['mesh']['cellX']
              faces_old, centers_old, levels_old, y_old, nodes_old = node1.getMesh()
              y_old_reshaped = y_old.T.reshape((-1,), order='F')
              
              print('\trefinement {}: {} --> {} pts'.format(itr,
                                                           old_old_centers.size,
                                                           centers_old.size))
  
              # update mesh structure
              options['mesh'] = setupFiniteVolumeMesh(xfaces=faces_old,
                                                      meshoptions=options['mesh'])
              if debug:
                plt.figure()
                plt.semilogy(old_old_centers, dx_opt_centers, label='optimal mesh')
                plt.semilogy(old_old_centers, dx_centers_old, label='previous mesh')
                plt.semilogy(centers_old, options['mesh']['cellSize'], label='new mesh')
                plt.grid()
                plt.legend()
                plt.title(f'Mesh at time t={t_old}, itr={itr}')
                plt.show()
  
        # Compute CFL time step
        dt = min( (t_span[-1]-t_old, cfl*getCFLtimestep(t_old,y_old_reshaped, options)) )
        if abs(last_out_t - t_old) > log_every:
          logger(f't={t_old:.3e}, dt={dt:.15e}')
          last_out_t = t_old
          
        # Perform step
        try:
          if 0: # high-order SSP method
            y_new_reshaped, _nfev = rk_nssp_53( f=lambda t,x: fun(t=t, x=x, options=options),
                                      tn=t_old, un=y_old_reshaped, dt=dt, limitfun=limitfun )
          else: # simple explicit Euler step
            y_new_reshaped = y_old_reshaped + dt*fun(t=t_old, x=limitfun(y_old_reshaped), options=options)
            _nfev = 1
          
          y_new = y_new_reshaped.reshape((3,-1), order='F').T
          t_new = t_old + dt
          nfev += _nfev
          if np.any(np.isnan( y_new_reshaped )):
            import pdb; pdb.set_trace()
            fun(t=t_old, x=limitfun(y_old_reshaped), options=options)
        # except ValueError as e:
        #   logger('issue: ', e)
        #   import pdb; pdb.set_trace()
        #   raise e
        #   break
        except Exception as e:
          logger('More problematic issue: ', e)
          import traceback
          traceback.print_exc()
          import pdb; pdb.set_trace()
          raise e
          break
  
        # Store states
        thist.append(t_new)
        yhist.append(y_new)
        centers_hist.append(centers_old)
        faces_hist.append(options['mesh']['faceX'].copy())
        levels_hist.append(levels_old)
        
        y_old = y_new      
        t_old = t_new
        nt += 1
                
        if nmax_step<nt:
          logger('maximum number of steps reached')
          break
    except KeyboardInterrupt:
      print('Early exit (user got bored and aborted ?)')
      pass
    
    out = OdeResult()
    out.status = 0
    if not np.allclose(thist[-1],t_span[-1], rtol=1e-10, atol=1e-10):
      logger(' /!\ end point not reached')

    out.nfev = nfev
    out.t = np.array(thist)
    out.y = yhist
    out.mesh_centers = centers_hist
    out.mesh_faces = faces_hist
    out.mesh_levels = levels_hist
    out.message = 'success'
    out.success = True
    out.sol = None

    return out
  
#%%
if __name__=='__main__':
  
  from Euler_FV_scheme import computeT, computeE, computeP, getXFromVars, getVarsFromX
  from Euler_FV_scheme import cv ,r, gamma
  from Euler_FV_scheme import modelfun
  

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
  length = 2
  x_interval = [-length/2, length/2]
  
  # mesh adaptation parameters
  dx_min = length/3e4
  dx_max = length/10
  # dx = length/(2**p)
  min_level = np.ceil( np.log(length/dx_max)/np.log(2) ).astype(int)
  max_level = np.ceil( np.log(length/dx_min)/np.log(2) ).astype(int)
  options['mesh']['adapt']={"mode":1,
                            "atol":1e-2, "rtol":1e-2}
  
  # Specifiy the physical duration of the simulation
  tend= 0.05
  
  
  # Generate the initial condition
  def init_fun(x):
    assert np.size(x) == 1
    xc = 0.
    if x < xc:
      P_0   =  1.*1e5
      rho_0 =  1.0
      u_0   =  0.0
    else:
      P_0   =  0.1*1e5
      rho_0 =  0.125
      u_0   =  0.0
    T_0 = computeT(P_0, rho_0)
    E_0 = cv*T_0 + 0.5*u_0*u_0
    return np.array((rho_0 , rho_0*u_0, rho_0*E_0))
  
  
  # compute evolution
  out = space_adapt_integration(fun=modelfun, t_span=(0., tend), cfl=0.4,
                                init_fun=init_fun, x_interval=x_interval,
                               options=options, level0=min_level, max_level=max_level,
                               nmax_step=np.inf, refine_every=1,
                               limitVars=True,events=None,
                               logger=print, log_every=0.)
  
  #%% Post-processing for secondary variables
  out.rho  = [y[:,0] for y in out.y]
  out.rhoU = [y[:,1] for y in out.y]
  out.rhoE = [y[:,2] for y in out.y]
  
  out.u, out.P, out.T, out.E, out.H, out.M = [], [], [], [], [], []
  for it in range(len(out.y)):
    temp = computeOtherVariables(out.rho[it], out.rhoU[it], out.rhoE[it])
    out.T.append( temp['T'] )
    out.P.append( temp['P'] )
    out.u.append( temp['u'] )
    out.E.append( temp['E'] )
    out.M.append( temp['M'] )
    out.H.append( temp['H'] )
    
  #%% Compute numerical Schlieren
  out.schlieren = []
  for it in range(len(out.y)):
                # (np.log10(np.abs(schlieren)), 'schlieren', 'Synthetic Schlieren', None, None, 'Greys'),
    out.schlieren.append( np.gradient((out.P[it]/out.rho[it]), out.mesh_centers[it]) )
  
  out.log10schlieren = [np.log10(np.abs(a)) for a in out.schlieren]
  
  #%% Plot
  tmaxplot = 25e-3
  # tmaxplot = out.t[-1]

  import matplotlib as mpl
  from matplotlib import cm
  step = out.t.size//1000
  for var, varname, clabel, lims, cmap in (
          # (out.rho,       'Density', r'$\rho$ (kg/m$^3$)', [0,1],        None),
          (out.T,         'T',       r'$T$ (K)',           [None, None], 'hot'),
          # (np.array(out.P)/1e5,     'P',       r'$P$ (bar)',        [None, None], None),
          # (out.u,         'Velocity',r'$u$ (m/s)',         [None, None], None),
          # (out.M, 'Mach', 'M', [-1, 1], 'seismic'),
          # (out.schlieren, 'Schlieren', None, [None, None], cm.binary),
          # (out.log10schlieren, 'Schlieren', None, [1, None], cm.binary),
          # (out.mesh_levels,    'Mesh refinement levels', 'level', [None, None], None),
        ):
    plt.figure(dpi=300)
    if lims[0] is None:
      lims[0] = min([min(a) for a in var])
    if lims[1] is None:
      lims[1] = max([max(a) for a in var])
    for it in tqdm(range(0, out.t.size-1, step)):
      plt.pcolormesh(out.mesh_faces[it],
                     [out.t[it], out.t[min(out.t.size-1, it+step)]],
                     [var[it]],
                     vmin=lims[0], vmax=lims[1], cmap=cmap)
      if it==0:
        if not (clabel is None):
          plt.colorbar(label=clabel)
        
      #edgecolors='black')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.ylim(0, tmaxplot)
    plt.title(varname)
    if len(varname)>1:
      varname = varname.lower()
    plt.savefig('{}_mod{}.png'.format(varname.replace(' ','_'),
                                      options['mesh']['adapt']["mode"]),
                dpi=300)
    plt.show()
  
  
  #%% Plot mesh statistics
  plt.figure()
  plt.plot(out.t, [a.size for a in out.mesh_centers])
  plt.grid()
  plt.xlim(0, tmaxplot)
  plt.ylim(0, None)
  plt.xlabel('t')
  plt.ylabel('mesh size')
  plt.title('Mesh size evolution')
  plt.savefig('mesh_size_mod{}.png'.format(options['mesh']['adapt']["mode"]), dpi=300)
  
  #%% Plot level population
  plt.figure()
  m2 = max([max(a) for a in out.mesh_levels])
  m1 = min([min(a) for a in out.mesh_levels])
  for ilevel in range (m1, m2+1):
    count = [np.count_nonzero(a==ilevel) for a in out.mesh_levels]
    plt.semilogy(out.t, count, label=f'{ilevel}')
  plt.grid()
  plt.legend()
  plt.xlim(0, tmaxplot)
  plt.ylim(0, None)
  plt.xlabel('t')
  plt.ylabel('mesh size')
  plt.title('Mesh distribution per level')
  plt.savefig('mesh_level_pop_mod{}.png'.format(options['mesh']['adapt']["mode"]), dpi=300)
  
  #%% Plot compression ratio
  plt.figure()
  max_level_each_step = np.array([np.max(a) for a in out.mesh_levels])
  uniform_nb_pts = length/(2**max_level_each_step)
  plt.semilogy(out.t, [uni / a.size for (a,uni) in zip(out.mesh_centers, uniform_nb_pts)])
  plt.grid()
  plt.xlim(0, tmaxplot)
  plt.xlabel('t')
  plt.ylabel('')
  plt.title('Mesh compression (vs uniform)')
  plt.savefig('mesh_compression_mod{}.png'.format(options['mesh']['adapt']["mode"]))
  
  #%% Total energy
  plt.figure()
  plt.semilogy(out.t, [sum(rhoE*np.diff(xfaces)) for (rhoE,xfaces) in zip(out.rhoE, out.mesh_faces)])
  plt.grid()
  plt.xlim(0, tmaxplot)
  plt.xlabel('t')
  plt.ylabel('Total energy')
  plt.title('Evolution of the overall energy')
  # plt.savefig('mesh_compression_mod{}.png'.format(options['mesh']['adapt']["mode"]))
  

  #%% Compare with analytical solution
  # /!\ The analytical solution assumes the initial discontinuity is at x=0
  from Euler_FV_scheme import Riemann_exact_from_conserved
  # for it in range(0,out.t.size,100):
  for it in [1000]:
    ysol = out.y[it]
    time = out.t[it]-out.t[0]
    mesh=out.mesh_centers[it]
    mesh_exact = np.linspace(np.min(mesh), np.max(mesh), int(2e3))
    
    exactsol = Riemann_exact_from_conserved(t=time, g=gamma,
                             WL=init_fun(-1),
                             WR=init_fun(1),
                             grid=mesh_exact)
    rho_exact = exactsol[0]
    u_exact = exactsol[1]
    P_exact = exactsol[2]
    T_exact = P_exact/rho_exact/r
  
  
    plt.figure()
    plt.plot(mesh_exact, rho_exact, color='r', label='exact')
    plt.scatter(mesh, ysol[:,0], color='b', label='simulation', marker='.')
    plt.xlabel('x (m)')
    plt.ylabel(r'$\rho$ (kg.m$^{-3}$)')
    plt.title('Density')
    plt.suptitle(f't={time:.3e}')
    plt.xlim(x_interval[0], x_interval[1]); plt.ylim(0,1.1); plt.grid()
    plt.legend()
    plt.savefig('analytical_comparison_mod{}.png'.format(options['mesh']['adapt']["mode"]), dpi=300)

 