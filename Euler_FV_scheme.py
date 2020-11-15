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
    assert not any(meshoptions['cellSize']==0.), 'some cells are of size 0...'
    assert not any(meshoptions['cellSize']<0.), 'some cells are of negative size...'
    assert not any(meshoptions['dxBetweenCellCenters']==0.), 'some cells have the same centers...'
    assert np.max(meshoptions['cellSize'])/np.min(meshoptions['cellSize']) < 1e10, 'cell sizes extrema are too different'
    # conveniency attributes for backward-compatibility  wtih finite-difference results post-processing
    meshoptions['x']  = meshoptions['cellX']
    meshoptions['dx'] = meshoptions['dxBetweenCellCenters']
    return meshoptions
  
def interfaceRiemannExact(WL,WR,options):
    """ Interface pour utiliser le solveur généreusement fourni par Storcky """
    # W = [rho rhoU, rhoE]
    rho, rhoU, rhoE = WL[0], WL[1], WL[2]
    temp = computeOtherVariables(rho, rhoU, rhoE)
    Wl_adapt = [rho, temp['u'], temp['P']]

    rho, rhoU, rhoE = WR[0], WR[1], WR[2]
    temp = computeOtherVariables(rho, rhoU, rhoE)
    Wr_adapt = [rho, temp['u'], temp['P']]

    # gives the exact solution of the riemann problem at x=0 and t=0
    rho,u,P =  Riemann_exact(t=0., g=gamma, Wl=Wl_adapt, Wr=Wr_adapt, grid=np.array([0.]))
    star_state = np.array([rho[0], rho[0]*u[0], 0.5*u[0]*u[0] + P[0]/((gamma-1)*rho[0])]) #same variables as input
    return star_state

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
    face_flux = np.zeros_like(WL)

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
    if total != SR.size: # some faces have not be solved
#    if np.isnan(SR+SL+Sstar).any():
        raise Exception('problem HLL UNRESOLVED CASE')

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


def modelfun(t,x,options):
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
    if np.isnan(time_deriv).any():
        raise Exception('NaNs in time_deriv, at time t={}'.format(t))

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

