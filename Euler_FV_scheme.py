#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 14:23:34 2019

One-dimensional Euler solver, with different spatial schemes and flux estimation methods.

@author: lfrancoi
"""

import numpy as np
import matplotlib.pyplot as plt

gamma= 1.4
R = 8.314
M = 28.967e-3 # masse molaire de l'air
r = R/M
cv = r/(gamma-1) # capacité thermique à volume constant
xorder = 0 # ordering of the state vector (0: cell by cell, 1: variables by variables)
           # xorder=0 leads to block-diagonal jacobian, better suited for implicit time integration

def plotfuncustom(x,y,time,name,selected_time_indices, marker=None):
    """ Automated line plots of the solution y at multiple instants on mesh x """
    plt.figure()
    colors=plt.cm.hsv(np.linspace(0,1,len(selected_time_indices)+2))
    for i in range(len(selected_time_indices)):
        plt.plot(x, y[selected_time_indices[i],:], label='t={}'.format(time[selected_time_indices[i]]), color=colors[i,:], marker=marker)
    plt.xlabel('x')
    plt.ylabel(name)
    if len(selected_time_indices)<7: #else the legend is too big
        plt.legend()
    plt.grid(None,which='minor',axis='both')
    plt.title(name)
    
def setupFiniteVolumeMesh(xfaces, meshoptions=None):
    """ Setup 1D spatial mash for finite volume, based on the positions of the faces of each cell """
    if meshoptions is None:
        meshoptions={}
    meshoptions['faceX'] = xfaces
    meshoptions['cellX'] = 0.5*(xfaces[1:]+xfaces[0:-1]) # center of each cell
    meshoptions['dxBetweenCellCenters'] = np.diff(meshoptions['cellX']) # gap between each consecutive cell-centers
    meshoptions['cellSize'] = np.diff(xfaces) # size of each cells
    meshoptions['nx'] = xfaces.size-1 # size of each cells
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
    # conveniency attributes for backward-compatibility  wtih finite-difference results post-processing
    meshoptions['x']  = meshoptions['cellX']
    meshoptions['dx'] = meshoptions['dxBetweenCellCenters']
    
    assert np.all( meshoptions['cellSize'] > 0 )
    assert np.all( meshoptions['dxBetweenCellCenters'] > 0 )
    return meshoptions
  
def interfaceRiemannExact(WL,WR,options):
    """ Interface pour utiliser le solveur généreusement fourni par Storcky """
    # W = [rho rhoU, rhoE]
    # gives the exact solution of the riemann problem at x=0 and t=0
    if WL.ndim==1: # single face evaluation
      rho,u,P =  Riemann_exact_from_conserved(t=0.,g=gamma,WL=WL,WR=WR,grid=np.array([0.]))
      star_state = np.array([rho[0], rho[0]*u[0], 0.5*u[0]*u[0] + P[0]/((gamma-1)*rho[0])]) #same variables as input
    else:
      star_state = np.zeros_like(WL)
      npts = WL.shape[1] 
      for i in range(npts):
        rho,u,P =  Riemann_exact_from_conserved(t=0.,g=gamma,WL=WL[:,i],WR=WR[:,i],grid=np.array([0.]))
        star_state[:,i] = np.array([rho[0], rho[0]*u[0], 0.5*u[0]*u[0] + P[0]/((gamma-1)*rho[0])])
    return star_state

def Riemann_exact_from_conserved(t,g,WL,WR,grid):
    """ Interface for use with [rho, rho*u, rho*E] """
    rho, rhoU, rhoE = WL[0], WL[1], WL[2]
    temp = computeOtherVariables(rho, rhoU, rhoE)
    Wl_adapt = [rho, temp['u'], temp['P']]

    rho, rhoU, rhoE = WR[0], WR[1], WR[2]
    temp = computeOtherVariables(rho, rhoU, rhoE)
    Wr_adapt = [rho, temp['u'], temp['P']]

    # gives the exact solution of the riemann problem at x=0 and t=0
    return  Riemann_exact(t=t, g=gamma, Wl=Wl_adapt, Wr=Wr_adapt, grid=grid)

def Riemann_exact(t,g,Wl,Wr,grid):
    """ Computes the exact solution of the Riemann problem with the state
        vectors W = [rho u P] at the left and right of the discontinuity.
        The initial discontinuity is located at x=0
        Other arguments:
            t - time
            g - gamma
            grid - grid over which to solve the problem"""
    # Solve for p2/p1
    r41 = Wl[2]/Wr[2]
    c1  = np.sqrt(g*Wr[2]/Wr[0])
    c4  = np.sqrt(g*Wl[2]/Wl[0])
    # f_p2p1 = lambda r,Wl,Wr,g,c1,c4,r41: r41-r* np.power(1+((g-1)/(2*c4))*(Wl[1]-Wr[1]-(c1/g)*((r-1)/np.sqrt(1+(r-1)*((g+1)/(2*g))))),(-2*g/(g-1)))
    # opt_fun = lambda x: f_p2p1(x,Wl,Wr,g,c1,c4,r41)
    # out = scipy.optimize.brentq(opt_fun, 0., 1e1, args=(), xtol=1e-10, rtol=1e-10, maxiter=100, full_output=False, disp=True)
    # r21 = out
    # out = scipy.optimize.newton(func=opt_fun, x0=3., fprime=None, args=(), tol=1.48e-08, maxiter=50, fprime2=None, x1=None, rtol=1e-10, full_output=False, disp=True)
    # r21 = out.x
    f_p2p1 = lambda r,Wl,Wr,g,c1,c4,r41: np.abs( r41-r* np.power(1+((g-1)/(2*c4))*(Wl[1]-Wr[1]-(c1/g)*((r-1)/np.sqrt(1+(r-1)*((g+1)/(2*g))))),(-2*g/(g-1))))
    import scipy.optimize
    out = scipy.optimize.minimize(fun = lambda x: f_p2p1(x,Wl,Wr,g,c1,c4,r41), x0=3, bounds=((0,np.inf),) )
    r21 = out.x[0]

    # Compute the remainder of properties in region 2 (post-shock)
    v2 = Wr[1] + (c1/g)*(r21-1)/np.sqrt(1+((g+1)/(2*g))*(r21-1))
    c2 = c1*np.sqrt(r21*(r21+((g+1)/(g-1)))/(1+r21*((g+1)/(g-1))))
    V = Wr[1] + c1*np.sqrt(1+(r21-1)*(g+1)/(2*g))
    P2 = r21*Wr[2]
    rho2 = g*P2/np.power(c2,2)

    # Determine the properties fo the flow in region 3
    v3 = v2
    P3 = P2
    c3 = c4 + (g-1)*(Wl[1]-v3)/2
    rho3 = g*P3/np.power(c3,2)

    # Find the boundaries of each region
    x1 = V*t
    x2 = v3*t
    x3 = (v3-c3)*t
    x4 = (Wl[1]-c4)*t

    # Compute the values of the state vector in regions 1 to 4
    P = Wr[2]*(grid > x1) + P2*(grid <= x1)*(grid > x2) + P3*(grid <= x2)*(grid > x3) + Wl[2]*(grid <= x4)
    rho = Wr[0]*(grid > x1) + rho2*(grid <= x1)*(grid > x2) + rho3*(grid <= x2)*(grid > x3) + Wl[0]*(grid <= x4)
    u = Wr[1]*(grid > x1) + v2*(grid <= x1)*(grid > x2) + v3*(grid <= x2)*(grid > x3) + Wl[1]*(grid <= x4)

    # Do the same in the expansion fan
    if t!=0:
        grid_over_t = grid/t
    else:
        grid_over_t = np.sign(grid) # pour que les résultats restent bons
    u = u + (grid <= x3)*(grid > x4)*(2*((grid_over_t)+c4+(g-1)*Wl[1]/2)/(g+1))
    cfan = (grid <= x3)*(grid > x4)* (2*((grid_over_t)+c4+(g-1)*Wl[1]/2)/(g+1)-grid_over_t)
    P = P + (grid <= x3)*(grid > x4)*Wl[2]*np.power(cfan/c4, 2*g/(g-1))
    rho[np.where((grid <= x3) & (grid > x4))] = g*P[np.where((grid <= x3) & (grid > x4))] / np.power(cfan[np.where((grid <= x3) & (grid > x4))], 2)
    return rho,u,P

def computeT(P,rho):
    return P/(rho*r)

def computeP(rho,T):
    return rho*r*T

def computeRho(P,T):
    return P/(r*T)

def computeE(T,u):
    return cv*T + 0.5*u*u
  
def computeOtherVariables(rho, rhoU, rhoE):
    u = rhoU/rho
    # E = cv*T + 0.5*u^2
    E = rhoE/rho
    T = (E - 0.5*u*u)/cv
    a = np.sqrt(gamma*r*T)
    P = computeP(rho, T)
    H = 0.5*u*u + a*a/(gamma-1) #E + P/rho
    M = u/a
    return {'u':u,'T':T, 'P':P, 'H':H, 'E':E, 'a':a, 'M':M}

def fluxEulerPhysique(W):
    """ Physical Euler fluxes """
    if len(W.shape)<2:
        W = W.reshape(W.shape[0],1)
        bReshaped=True
    else:
        bReshaped=False
    rho = W[0,:]
    rhoU = W[1,:]
    rhoE = W[2,:]

    out = computeOtherVariables(rho, rhoU, rhoE)
    u,P = out['u'],out['P']

    F = np.zeros_like(W)
    F[0,:] = rhoU
    F[1,:] = rhoU*u + P
    F[2,:] = rhoE*u + P*u

    if bReshaped:
        return F.reshape((F.size,))
    else:
        return F


def HLLC_solver(WL,WR,options):
    """ HLLC approximate Riemann solver, as described in Toro's boo"""
    if len(WR.shape)<2:
        WR = WR.reshape(WR.shape[0],1)
        WL = WL.reshape(WL.shape[0],1)
        bReshaped=True
    else:
        bReshaped=False
    # 1 - compute physical variables
    rhoL, rhoUL,rhoEL = WL[0,:], WL[1,:], WL[2,:]
    rhoR, rhoUR,rhoER = WR[0,:], WR[1,:], WR[2,:]
    out = computeOtherVariables(rhoR, rhoUR, rhoER)
    uR,PR,ER,HR,aR = out['u'],out['P'], out['E'], out['H'], out['a']
    out = computeOtherVariables(rhoL, rhoUL, rhoEL)
    uL,PL,EL,HL,aL = out['u'],out['P'], out['E'], out['H'], out['a']

    # compute fluxes
    face_flux = np.zeros_like(WL)*np.nan # the NaN initialisation allows to spot issues easily

    # vectorized mode
    # estimate the wave speeds
    if 0: #based on Roe-average
        utilde = (np.sqrt(rhoL)*uL + np.sqrt(rhoR)*uR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
        Htilde = (np.sqrt(rhoL)*HL + np.sqrt(rhoR)*HR)/(np.sqrt(rhoL) + np.sqrt(rhoR))
        atilde = np.sqrt( (gamma-1)*(Htilde-0.5*utilde*utilde) )
        SL = utilde-atilde
        SR = utilde+atilde
    else:
        SL = np.minimum(uL-aL, uR-aR)
        SR = np.minimum(uL+aL, uR+aR)
    # /!\ In Toro's book, the energy "E" is actually our rhoE
    # compute Sstar
    Sstar = ( PR-PL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR) ) / ( rhoL*(SL-uL) - rhoR*(SR-uR) )
    Wstar_L = np.zeros_like(WL)
    Wstar_L[0,:] = rhoL*(SL-uL)/(SL - Sstar)
    Wstar_L[1,:] = rhoL*(SL-uL)/(SL - Sstar)*Sstar
    Wstar_L[2,:] = rhoL*(SL-uL)/(SL - Sstar)*( EL+ (Sstar-uL)*(Sstar + PL/(rhoL*(SL-uL))) )

    Wstar_R = np.zeros_like(WL)
    Wstar_R[0,:] = rhoR*(SR-uR)/(SR - Sstar)
    Wstar_R[1,:] = rhoR*(SR-uR)/(SR - Sstar)*Sstar
    Wstar_R[2,:] = rhoR*(SR-uR)/(SR - Sstar)*( ER+ (Sstar-uR)*(Sstar + PR/(rhoR*(SR-uR))) )

    total=0
    I=np.where(SL>0)
    face_flux[:,I] = fluxEulerPhysique(WL[:,I])
    total = total + np.size(I)

    I=np.where((SL<=0) & (Sstar>=0))
    face_flux[:,I] = fluxEulerPhysique(Wstar_L[:,I])
    total = total + np.size(I)

    I=np.where((SR>0) & (Sstar<0))
    face_flux[:,I] = fluxEulerPhysique(Wstar_R[:,I])
    total = total + np.size(I)

    I = np.where(SR<=0)
    face_flux[:,I] = fluxEulerPhysique(WR[:,I])
    total = total + np.size(I)
#     if total != SR.size: # some faces have not be solved
# #    if np.isnan(SR+SL+Sstar).any():
#         raise Exception('problem HLL UNRESOLVED CASE')

    if bReshaped:
        return face_flux.reshape((face_flux.size,))
    else:
        return face_flux
     
      
def FVS_solver(WL,WR,options):
    """ Flux Vector Splitting Riemann solver, using the Stegar-Warming splitting """
    if len(WR.shape)<2:
        WR = WR.reshape(WR.shape[0],1)
        WL = WL.reshape(WL.shape[0],1)
        bReshaped=True
    else:
        bReshaped=False
    # compute physical variables
    rhoL, rhoUL,rhoEL = WL[0,:], WL[1,:], WL[2,:]
    rhoR, rhoUR,rhoER = WR[0,:], WR[1,:], WR[2,:]
    out = computeOtherVariables(rhoR, rhoUR, rhoER)
    uR,PR,ER,HR,aR = out['u'],out['P'], out['E'], out['H'], out['a']
    out = computeOtherVariables(rhoL, rhoUL, rhoEL)
    uL,PL,EL,HL,aL = out['u'],out['P'], out['E'], out['H'], out['a']

    #### compute fluxes
    face_flux = np.zeros_like(WL)  
    epsilon = 1e-2  # to smooth the eigenvalues, don't remember why :\
    # Notations are coherent with Toro's book (page 276)
    # For each cell, the flux is Fmoins(right_cell) + Fright(left_cell)
    # 1 - Determine F+ from the left cell of each itnerface
    Fplus = np.zeros(WL.shape)
    rho,a,u,H = rhoL, aL, uL, HL
    valeurs_propres = np.zeros_like(face_flux)
    valeurs_propres[0,:] = u-a
    valeurs_propres[1,:] = u
    valeurs_propres[2,:] = u+a
    lbda_plus    = (valeurs_propres + np.sqrt(valeurs_propres*valeurs_propres + epsilon**2))*0.5
    #            lbda_plus    = (valeurs_propres + np.abs(valeurs_propres))/2
    Fplus[0,:] = rho/(2*gamma) * (         lbda_plus[0,:] +   2*(gamma-1)*lbda_plus[1,:] +         lbda_plus[2,:] )
    Fplus[1,:] = rho/(2*gamma) * (   (u-a)*lbda_plus[0,:] + 2*(gamma-1)*u*lbda_plus[1,:] +   (u+a)*lbda_plus[2,:] )
    Fplus[2,:] = rho/(2*gamma) * ( (H-u*a)*lbda_plus[0,:] + (gamma-1)*u*u*lbda_plus[1,:] + (H+u*a)*lbda_plus[2,:] )

    # 2 - determine F- from the right cell of each interface
    Fmoins    = np.zeros(WL.shape)
    rho,a,u,H   = rhoR, aR, uR, HR
    valeurs_propres = np.zeros_like(face_flux)
    valeurs_propres[0,:] = u-a
    valeurs_propres[1,:] = u
    valeurs_propres[2,:] = u+a
    #            lbda_moins   = (valeurs_propres - np.abs(valeurs_propres))/2
    lbda_moins   = (valeurs_propres - np.sqrt(valeurs_propres*valeurs_propres + epsilon**2))*0.5
    Fmoins[0,:] = rho/(2*gamma) * (        lbda_moins[0,:] +   2*(gamma-1)*lbda_moins[1,:] +         lbda_moins[2,:] )
    Fmoins[1,:] = rho/(2*gamma) * (  (u-a)*lbda_moins[0,:] + 2*(gamma-1)*u*lbda_moins[1,:] +   (u+a)*lbda_moins[2,:] )
    Fmoins[2,:] = rho/(2*gamma) * ((H-u*a)*lbda_moins[0,:] + (gamma-1)*u*u*lbda_moins[1,:] + (H+u*a)*lbda_moins[2,:] )

    # 3 - sum F+ and F- to get the actual flux going through each interface
    face_flux[:,:]   = Fmoins + Fplus
    
    if bReshaped:
        return face_flux.reshape((face_flux.size,))
    else:
        return face_flux
      
      
def naive_centered_solver(WL,WR,options):
  """ centered scheme with artifical dissipation """
  W=(WR+WL)/2
  face_flux = fluxEulerPhysique(W)
  
  # add artificial dissipation
  dWdx_faces = np.zeros_like(W)
  xfaces = options['mesh']['faceX']
  xgaps = options['mesh']['dxBetweenCellCenters']
  for i in range(3):
      dWdx_faces[i,1:-1] = (W[i,1:]-W[i,:-1])/xgaps
      dWdx_faces[i,0] = (W[i,0]-W[i,1])/(xfaces[0]-xfaces[1])
      dWdx_faces[i,-1] = (W[i,-1]-W[i,-2])/(xfaces[-1]-xfaces[-2])
  D = 2*1e1
  dissipation_flux = -D*dWdx_faces
  face_flux = face_flux + dissipation_flux
  return face_flux
  

def naive_upwind_solver(WL,WR,options):
  """ naive solver that returns the correct flux only if the flow is supersonic
  toards the right --> should only be used for debur purposes """
  return fluxEulerPhysique( WL )


def minmax_limiter(Wnew,WR,WL):
  """ Min/Max limiter --> the reconstructed states at the interface will no be
  higher or lower than the ones of the adjacent cell """
  maxies = np.maximum(WR,WL)
  minies = np.minimum(WR,WL)
  Wnew = np.minimum(maxies,Wnew)
  Wnew = np.maximum(minies,Wnew)
  return Wnew


def modelfun(t,x,options, estimate_space_error=False):
    """ ODE function for Euler equation
    Inputs:
      t: float
        time of the evaluation. May be used to compute specific forcing terms
      y: array-like
        current solution vector
    Output:
      dydt: array-like
        time derivatives of the solution components
    """
    print(t)

    # gather mesh information
    xcells = options['mesh']['cellX']
    xfaces = options['mesh']['faceX']
    xgaps = options['mesh']['dxBetweenCellCenters']
    nx = xcells.size

    # Compute conserved variables
    rho, rhoU, rhoE = getVarsFromX(x, options)
    temp = computeOtherVariables(rho, rhoU, rhoE)
    u = temp['u']
    P = temp['P']

    # Select the Riemann solver
    nMode = options['scheme']['riemann_solver']
    if nMode==1:
      solver = naive_upwind_solver
    if nMode==1.5:
      solver = naive_centered_solver
    elif nMode==2:
      solver = FVS_solver
    elif nMode==3:
      solver = HLLC_solver
    elif nMode==4: # exact Riemann solver (not optimised, very slow)
      solver = interfaceRiemannExact
    else:
      raise Exception('Unknown solver mode {}'.format(nMode))

    # Select the limiter (for high-order recosntruction of the interface values)      
    nLimiter = options['scheme']['limiter']
    if nLimiter==0:
        limiter = lambda Wnew,WR,WL: Wnew
    elif nLimiter==1: # min,max
        limiter = minmax_limiter
    else:
        raise Exception('unknown limiter mode {}'.format(nLimiter))
        
    # Compute the "left" and "right" states for each (inter)face
    WL = np.zeros((3,nx+1)) # left state
    WR = np.zeros((3,nx+1)) # right state

    W = np.vstack((rho, rhoU, rhoE)) #TODO: check
    nReconstruction = options['scheme']['reconstruction']
    if nReconstruction==0: # order 1, all the variables are assumed constant within each cell
        WL[:,1:] = np.vstack((rho, rhoU, rhoE))
        WR[:,:-1] = WL[:,1:]
    elif nReconstruction==1: # 2nd order MUSCL
        # non-TVD MUSCL-Hancock (see Toro page 505)
        # compute slopes
        delta_faces = np.zeros( (W.shape[0], W.shape[1]+1) )
        delta_faces[:,1:-1] = (W[:,1:]-W[:,:-1])/xgaps

        # slopes at the center of each cell
        omega = 0.5 #should be in [-1,1]
        delta_cells = 0.5*(1+omega)*delta_faces[:,:-1] + 0.5*(1-omega)*delta_faces[:,1:]

        # calcul des valeurs aux faces
        for i in range(3):
            # /!\ We don't use the differences as in Toro's book, but rather the gradients
            # --> better for non-uniform meshes
            WL[i,1:]  =  W[i,:] + delta_cells[i,:]*(xcells-xfaces[:-1])
            WR[i,:-1] =  W[i,:] + delta_cells[i,:]*(xfaces[1:]-xcells)
            
            # states at the boundaries
            WR[i,0]  = W[i,0]  + delta_cells[i,0 ]*(xfaces[0]-xcells[0])
            WL[i,-1] = W[i,-1] + delta_cells[i,-1]*(xfaces[-1]-xcells[-1])
    else:
      raise Exception('Unknown recosntruction type {}'.format(nReconstruction))
      
    # COnstruct ghost states for the boundaries
    if options['BCs']['left']['type']=="transmissive":
        ghost_state_L = W[:,0]
    elif options['BCs']['left']['type']=="reflective":
        ghost_state_L = np.array([ W[0,0], -W[1,0], W[2,0]]) 
    else:
      raise Exception('Unknown BC type {} for left side'.format(options['BCs']['left']['type']))
      
    if options['BCs']['right']['type']=="transmissive":
        ghost_state_R = W[:,-1]
    elif options['BCs']['right']['type']=="reflective":
        ghost_state_R = np.array([ W[0,-1], -W[1,-1], W[2,-1]])
    else:
        raise Exception('Unknown BC type {} for right side'.format(options['BCs']['right']['type']))
        
    WL[:,0]  = ghost_state_L
    WR[:,-1] = ghost_state_R
    
    if estimate_space_error:
      if not'adapt' in options['mesh'].keys():
        raise Exception('no mesh adaptation parameters in options')
      spaceerrmode =  options['mesh']['adapt']['mode']
      
      # compute equivalent dx at each face
      dx_faces = np.zeros((nx+1,))
      dx_faces[0] = options['mesh']['cellSize'][0]
      dx_faces[-1] = options['mesh']['cellSize'][0]
      dx_faces[1:-1] = 0.5 * ( options['mesh']['cellSize'][1:] + options['mesh']['cellSize'][:-1])
      
      if spaceerrmode==0:
        raise Exception('error')
        
      else:  # estimate and direclty return an optimal cell size
        atol,rtol= options['mesh']['adapt']['atol'], options['mesh']['adapt']['rtol']
        if spaceerrmode==1: 
          # 1 - solve unlimited Riemman problems with centered scheme and compare error
          #     with actual scheme
          face_flux = solver(WL,WR,options)
          face_flux_centered = fluxEulerPhysique(W=(WL+WR)/2)
          err = abs(face_flux - face_flux_centered) / (atol+rtol*np.minimum(abs(face_flux_centered), abs(face_flux)))
          err = np.max(err, axis=0)
          dx_opt = dx_faces/np.sqrt(err)
          
        elif spaceerrmode==2:
          # 2 - simply compute the disagreeement etween the second-order reconstructions
          #     at the faces
          err = abs(WL - WR) / (atol+rtol*np.minimum(abs(WR), abs(WL)))
          err = np.max(err, axis=0)
          dx_opt = dx_faces/np.sqrt(err)
          
        elif spaceerrmode==3:
          # 2 - compare to first-order scheme
          face_flux = solver(WL,WR,options)
          
          WL1 = np.zeros((3,nx+1)) # left state
          WR1 = np.zeros((3,nx+1)) # right state
          WL1[:,0] = W[:,0]
          WL1[:,-1] = W[:,-1]
          WL1[:,1:] = np.vstack((rho, rhoU, rhoE))
          WR1[:,:-1] = WL1[:,1:]
          face_flux_1 = solver(WL1,WR1,options)
          err = abs(face_flux - face_flux_1) / (atol+rtol*np.minimum(abs(face_flux_1), abs(face_flux)))
          # the error is O(dx^2) with a MUSCL reconstruction
          # A * dx_opt^2 = 1
          # A * dx^2 = err > 1   => dx_opt/dx = sqrt(1/err)
          err = np.max(err, axis=0)
          dx_opt = dx_faces/np.sqrt(err)

        elif spaceerrmode==4:
          # 2 - gradient of density
          dx_faces = np.hstack((options['mesh']['cellX'][0]-options['mesh']['faceX'][0],
                                options['mesh']['dxBetweenCellCenters'],
                                options['mesh']['faceX'][-1]-options['mesh']['cellX'][-1],
                               ))
          dx_opt = np.zeros((nx+1),)
          for i in range(1,nx-1):
            v1 = rho[:-1]
            v2 = rho[1:]
            err = abs(v1-v2) / ( atol + rtol * np.minimum(abs(v1), abs(v2)) )
            dx_opt[1:-1] = dx_faces[1:-1] / err
          # boundaries
          v1 = WL[0,0]
          v2 = rho[0]
          err = abs(v1-v2) / ( atol + rtol * np.minimum(abs(v1), abs(v2)) )
          dx_opt[0] = dx_faces[0] / err
          
          v1 = WL[0,-1]
          v2 = rho[-1]
          err = abs(v1-v2) / ( atol + rtol * np.minimum(abs(v1), abs(v2)) )
          dx_opt[-1] = dx_faces[-1] / err
        
        else:
          raise Exception(f'Space error estimation method {spaceerrmode} does not exist')
      return dx_opt
    
    # Limit the states
    WL[:,1:-1] = limiter(WL[:,1:-1], W[:,0:-1], W[:,1:])
    # WL[:,0] and  WL[:,-1] are not limited
    WR[:,1:-1] = limiter(WR[:,1:-1], W[:,0:-1], W[:,1:])
    # WR[:,0] and WR[:,-1] are also not limited
    
    if 0: # debug plots for the reconstruction
      for i in range(3):
          plt.figure()
          plt.plot(WL[i,:], label='W', color='g', marker='+', linestyle='--')
          plt.plot(WR[i,:], label='WR', color='r', marker='.')
          plt.plot(WL[i,:], label='WL', color='b', marker='.')
          plt.legend()
          plt.title('Variable {} at t={}'.format(i, t))
          plt.show()
      raise Exception('debug MUSCL')
    
    # Compute the fluxes by solving a Riemann problem at each face
    face_flux = solver(WL,WR,options)

    # Compute time derivatives of the conserved variables
    time_deriv = (1/options['mesh']['cellSize']) * (face_flux[:,:-1] - face_flux[:,1:])
    time_deriv = getXFromVars(rho=time_deriv[0,:], rhoU=time_deriv[1,:], rhoE=time_deriv[2,:])
    # if np.isnan(time_deriv).any():
    #     raise Exception('NaNs in time_deriv, at time t={}'.format(t))

    #### Plots
    if 0:
        plt.figure()
        plt.plot(xcells, rhoU, label='rho*u', marker='o')
        plt.plot(xfaces, face_flux[0,:], label='mass flux', marker='+')
        plt.legend()
        plt.xlabel('x')
        plt.title('face flux (continuity) at t={}'.format(t))

        plt.figure()
        plt.plot(xcells, rhoU*u+P, label='rho*u^2 + P', marker='o')
        plt.plot(xfaces, face_flux[1,:], label='momentum flux', marker='+')
        plt.legend()
        plt.xlabel('x')
        plt.title('face flux (momentum) at t={}'.format(t))

        plt.figure()
        plt.plot(xcells, u*(rhoE+P), label='u*(rho*E+P)', marker='o')
        plt.plot(xfaces, face_flux[2,:], label='total energy flux', marker='+')
        plt.legend()
        plt.xlabel('x')
        plt.title('face flux (energy) at t={}'.format(t))
        raise Exception('debug stop')

    return time_deriv

def getXFromVars(rho, rhoU, rhoE):
    """ Constructs the state vector x from its components (conserved variables) """
    if xorder==0:
       if rho.ndim==1:
           return np.dstack((rho, rhoU, rhoE)).reshape((-1,), order='C')
       else: # time axis or perturbations
           return np.dstack((rho, rhoU, rhoE)).reshape((-1, rho.shape[2]), order='C')      
    else:
      return np.hstack((rho, rhoU, rhoE))

def getVarsFromX(x, options):
    """ Extracts the conserved variables from the state vector x """
    nx = options['mesh']['nx']
    if xorder==0:
      Xresh = x.reshape((nx,3))
      rho = Xresh[:,0]
      rhoU = Xresh[:,1]
      rhoE = Xresh[:,2]
    else:
      rho = x[:nx]
      rhoU = x[nx:2*nx]
      rhoE = x[2*nx:]
    return rho,rhoU,rhoE
  
def getVarsFromX_vectorized(x, options):
    """ Extracts the conserved variables from the state vector x, but
    here the function is adapted for the post-processing of the time history
    of the solution (1 slice = 1 time step) """
    nx = options['mesh']['nx']
    if xorder==0:
      Xresh = x.reshape((nx,3,-1))
      rho = Xresh[:,0,:]
      rhoU = Xresh[:,1,:]
      rhoE = Xresh[:,2,:]
    else:
      rho = x[:nx,:]
      rhoU = x[nx:2*nx,:]
      rhoE = x[2*nx:,:]
    return rho,rhoU,rhoE


def getCFLtimestep(t,x,options):
    """ Returns the CFL=1 time step """
    dx = options['mesh']['cellSize']
    ncells = dx.size

    ##### recover conserved variables
    rho, rhoU, rhoE = getVarsFromX(x,options)
    temp = computeOtherVariables(rho, rhoU, rhoE)
    u = temp['u']
    a = temp['a']
    dt = dx / np.maximum(np.abs(u+a),np.abs(u-a))
    return np.min(dt)

def limitValues(W):
    """ Limit conserved variables based on bounds on the intensive variables """
    PRESSURE_MIN = 10.
    T_MIN = 10.
    # compute current intensive variables
    rho, rhoU,rhoE = W[0,:], W[1,:], W[2,:]
    out = computeOtherVariables(rho=rho, rhoU=rhoU, rhoE=rhoE)
    u,P,E,H,a,T = out['u'], out['P'], out['E'], out['H'], out['a'], out['T']

    # find where bounds are violated and correct the intensive variables
    bModif = False
    I = np.where(P<PRESSURE_MIN)[0]
    if len(I)>0:
        P[I] = PRESSURE_MIN
        bModif = True
    I = np.where(T<T_MIN)[0]
    if len(I)>0:
        T[I] = T_MIN
        bModif = True

    if bModif:
        # compute corrected conserved variables
        rho = computeRho(P=P, T=T)
        rhoU = rho * u
        rhoE = rho * computeE(T=T,u=u)
        W[0,:], W[1,:], W[2,:] = rho, rhoU,rhoE
    return W
  
def limit_stages(y,options):
  W_new = np.vstack( getVarsFromX(y, options) ) 
  W_new = limitValues(W=W_new)
  y_new = getXFromVars( rho=W_new[0,:],
                        rhoU=W_new[1,:],
                        rhoE=W_new[2,:] )
  return y_new
  
def rk_nssp_53( f, tn, un, dt, limitfun=lambda y: y ):
    """RK NSSP (5,3)
        Runge-Kuta method with 5 stages, of order 3
        f     function f(t,u)
        tn    current time
        un    current solution at time tn
        dt    time step to the next time

        return unp1 solution at time tn


        source: https://josselin.massot.gitlab.labos.polytechnique.fr/ponio/viewer.html#rk_nssp_53
    """
    def lf(t,y):
      return f(t,limitfun(y))
    k1 = lf(tn, un)
    k2 = lf(tn+dt/7,    dt*k1/7 + un )
    k3 = lf(tn+3*dt/16, 3*dt*k2/16 + un)
    k4 = lf(tn+dt/3,    dt*k3/3 + un)
    k5 = lf(tn+2*dt/3,  2*dt*k4/3 + un)
    return limitfun( dt*(k1/4 + 3*k5/4) + un ), 5 # sol and fun evals

def CFLintegration(fun,t_span,cfl,y0,methodclass, max_step=np.inf,
                   nmax_step=np.inf, relvar_min=None, datalogger=None,
                   jacband=None, limitVars=False,events=None, options=None,
                   logger=print, log_every=0.):
    """ CFL-driven time integration
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
    # if callable(events):
    #     events = list(events)

    from scipy.integrate._ivp.ivp import OdeResult
    from scipy.integrate._ivp.ivp import handle_events, find_active_events, prepare_events

    t_old = t_span[0]
    y_old = y0.copy()
    yhist=[y_old]
    thist=[t_old]
    
    if limitVars:
      limitfun = lambda y: limit_stages(y=y, options=options)
    else:
      limitfun = lambda y: y

    if methodclass is None:
        logger('using custom method')
        class Method():
            def __init__(self):
                self.y=None
                self.f=None
                self.max_step = None
                self.first_step = None
                self.h_abs = None
                self.nfev = 0
                self.njev = 0
                self.nlu = 0
                self.t = None

            def step(self):
                dt = self.h_abs
                self.y_old = self.y
                self.y, nfev = rk_nssp_53( f=fun, tn=self.t, un=self.y, dt=dt, limitfun=limitfun )
                self.t += dt
                self.nfev += nfev

        method = Method()
        method.t = t_old
        method.y = y_old
        method.options=options
    else:
        logger('using Scipy solvers')
        # raise Exception('should not be used (lacks limitation)')
        method = methodclass(fun=fun, t0=t_span[0], y0=y0, t_bound=t_span[-1], max_step=np.inf,
                             jacband=jacband, uband=jacband, lband=jacband, limitfun=limitfun,
                             rtol=1e50, atol=1e50, vectorized=False, first_step=None,events=events)

    events, is_terminal, event_dir = prepare_events(events)
    terminate = False
    if events is not None:
        g = [event(t_old, y_old) for event in events]
        t_events = [[] for _ in range(len(events))]
        y_events = [[] for _ in range(len(events))]
        active_events = find_active_events(g, g, event_dir)
        if active_events.size > 0:
            logger('Event occured at t0 !')
            sol = lambda t: y_old
            t_new = t_old
            root_indices, roots, terminate = handle_events( sol, events, active_events, is_terminal, t_old, t_new)
            for e, te in zip(root_indices, roots):
                t_events[e].append(te)
                y_events[e].append(sol(te))
            if terminate:
                t_new = roots[-1]
                y_new = sol(t_new)
    else:
        t_events = None
        y_events = None

    last_out_t = t_old
    if not terminate:
      while t_old<t_span[-1]:
        dt = min( (t_span[-1]-t_old, cfl*getCFLtimestep(t_old,y_old, options)) )
        if abs(last_out_t - t_old) > log_every:
          logger(f't={t_old:.3e}, dt={dt:.15e}')
          last_out_t = t_old
        method.max_step=dt
        method.first_step=dt
        method.h_abs=abs(dt)
        method.h    =abs(dt)

        try:
          method.step()
        except ValueError as e:
          logger('issue: ', e)
          raise e
          break
        except Exception as e:
          logger('More problematic issue: ', e)
          break
        
        t_new = method.t
        y_new = method.y

        dt_eff = t_new-t_old
        if not np.allclose(dt_eff, dt, rtol=1e-2, atol=1e-12):
            logger('dt     =',dt)
            logger('dt_eff =', dt_eff)
            # raise Exception('CFL was not maintained')
        # if limitVars:
        #   # Limit values based on non-conserved variables
        #   y_new = limit_stages(y_new,options)
        ##   W_new = np.vstack( getVariablesFromX(y_new, options) ) 
        ##   W_new = limitValues(W=W_new, dxR=None, options=options)
        ##   y_new = constructXfromVariables( rho=W_new[0,:],
        ##                                    rhoU=W_new[1,:],
        ##                                    rhoE=W_new[2,:],
        ##                                    options=options )
        ##   method.y = y_new

        if events is not None:
            g_new = [event(t_new, y_new) for event in events]
            active_events = find_active_events(g, g_new, event_dir)
            # TODO: precise event occurnece as in solve_ivp ?
            # linear interpolation in the mean time...
            sol = lambda t: y_old + (y_new - y_old) * ( t - t_old ) / (t_new - t_old)
            if active_events.size > 0:
                root_indices, roots, terminate = handle_events( sol, events, active_events, is_terminal, t_old, t_new)
                for e, te in zip(root_indices, roots):
                    t_events[e].append(te)
                    y_events[e].append(sol(te))
                if terminate:
                    t_new = roots[-1]
                    y_new = sol(t_new)
                    logger(f'Event occured at t={t_new:.3e}')                    
            g = g_new
        if datalogger:
          datalogger.log(t_new, y_new)
        else: # backup of every step
          thist.append(t_new)
          yhist.append(y_new)
        
        if terminate:
            logger('Termination due to event')
            break
          
        if nmax_step<len(thist):
          logger('maximum number of steps reached')
          break
      
        if not (relvar_min is None):
          dvardt = (y_new-y_old) / dt_eff
          relvar = abs(dvardt/ (1e-9 + abs(y_old)) )  
          maxrelvar_step = np.max(relvar)
          if maxrelvar_step < relvar_min:
            logger('relative solution variations < {relvar_min:.2e} --> early exit')
            terminate = True
            break
          
        t_old = t_new
        y_old = y_new
        
    if datalogger: # add last time point anyway...
      thist.append(t_new)
      yhist.append(y_new)
    out = OdeResult()
    out.t_events = t_events
    out.y_events = y_events
    if not terminate:
        out.status = 0
        if not np.allclose(thist[-1],t_span[-1], rtol=1e-10, atol=1e-10):
          logger(' /!\ end point not reached')
    else:
        out.status = 1

    out.nfev = method.nfev
    out.njev = method.njev
    out.nlu  = method.nlu
    out.t = np.array(thist)
    out.y = np.array(yhist).T
    out.message = 'success'
    out.success = True
    out.sol = None

    return out