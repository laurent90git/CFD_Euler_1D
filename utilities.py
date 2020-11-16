#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:40:32 2019

@author: thomas
"""
import numpy as np
import copy
import collections
import math
pi=math.pi
import scipy.interpolate
import matplotlib.pyplot as plt
import json

class NumpyEncoder(json.JSONEncoder):
    """ Recursively converts numpy array into lists for JSON serialization """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def JSONtoNumpy(dico, key_hist=''):
  """ Recursively all list of floats (and list of list of floats) to numpy arrays in the input dictionnary """
  # print(key_hist)
  if isinstance(dico, dict):
    for key in dico.keys():
      dico[key] = JSONtoNumpy( dico[key], key_hist='.'.join( (key_hist, key) ) )
  else:
    if isinstance(dico, list):
      if len(dico)>0:
        if isinstance(dico[0], list) or isinstance(dico[0],dict):
          for i in range(len(dico)):
            dico[i] = JSONtoNumpy(dico[i], key_hist='.'.join( (key_hist, str(i))))
        elif isinstance(dico[0], np.ndarray):
          dico = np.array(dico)
        elif isinstance(dico[0], float):
          return np.array(dico)
  return dico

import os, shutil
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

### interacitive legends with mpld3
def interactive_legend2(ax=None, fig=None):
  import mpld3
  from mpld3 import plugins
  np.random.seed(9615)
  if ax is None:
    ax = plt.gca()
  if fig is None:
    fig = plt.gcf()
  handles, labels = ax.get_legend_handles_labels() # return lines and labels
  interactive_legend_plug = plugins.InteractiveLegendPlugin(zip(handles,
                                                           ax.collections),
                                                       labels,
                                                       alpha_unsel=0.2,
                                                       alpha_over=1.5,
                                                       start_visible=True)
  plugins.connect(fig, interactive_legend_plug)

def interactive_legend(ax=None):
    if ax is None:
        ax = plt.gca()
    if ax.legend_ is None:
        ax.legend()
    return InteractiveLegend(ax.get_legend())

class InteractiveLegend(object):
  ### issu de Stackoverflow, marche moyen
    def __init__(self, legend):
        self.legend = legend
        self.fig = legend.axes.figure

        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()

        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10) # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _build_lookups(self, legend):
        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:

            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def on_click(self, event):
        if event.button == 3:
            visible = False
        elif event.button == 2:
            visible = True
        else:
            return

        for artist in self.lookup_artist.values():
            artist.set_visible(visible)
        self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
            else:
                handle.set_visible(False)
        self.fig.canvas.draw()

    def show(self):
        plt.show()


def sigmoid_arctan(x, x0, width, n=1, p=1):
    # ancienne version
    # temp = np.abs(((x-x0)/width)**n)
    # return ( (np.arctan(temp)*2/pi*np.sign(x-x0)+1)/2 )**p
    return (0.5*( 1 + np.arctan(((x-x0)/width)**n)*2/np.pi ))**p

def define_my_arctansigmoid(T2, T1):
  # trouver les apramètres x0 et width tels que sigmoid_arctan(x=Tmax)=0.99
  # et sigmoid_arctan(x=Tmin)=0.01
  r1 = 0.01
  r2 = 0.99
  A1 = np.tan((r1-0.5)*np.pi/2)
  A2 = np.tan((r2-0.5)*np.pi/2)

  width = (T2-T1)/(A2-A1)
  x0 = 0.5*(T2+T1-width*(A1+A2))
  return width, x0

if __name__=='__main__':

  width, x0 = define_my_arctansigmoid(3000,2000)
  plt.figure()
  testx = np.linspace(-100, 5000,1000)
  # pltfun = plt.plot
  pltfun = plt.semilogy
  n=1
  p=1
  pltfun(testx, sigmoid_arctan(testx, x0=x0, width=width, n=n,p=p), label='fitted')
  pltfun(testx, sigmoid_arctan(testx, x0=x0, width=0.1, n=1,p=1), label='codée')
  # for w in np.logspace(-1,2,5):
  #   pltfun(testx, sigmoid_arctan(testx, x0=2500, width=w, n=n,p=p), label='w={:.2e}'.format(w))
  plt.legend()
  plt.title('test sigmoid arctan pout tfl cedre')

  plt.figure()
  testx = np.linspace(-100, 5000,1000)
  pltfun = plt.semilogy
  pltfun(testx, 1-sigmoid_arctan(testx, x0=x0, width=width, n=n,p=p), label='fitted')
  pltfun(testx, 1-sigmoid_arctan(testx, x0=x0, width=0.1, n=1,p=1), label='codée')
  for w in np.logspace(-1,2,5):
    pltfun(testx, 1-sigmoid_arctan(testx, x0=2500, width=w, n=n,p=p), label='w={:.2e}'.format(w))
  plt.legend()
  plt.title('test 1 - sigmoid arctan pout tfl cedre')


  plt.figure()
  testx = np.linspace(-1,1,1000)
  plt.plot(testx, sigmoid_arctan(testx,0,0.01), label='w=0.01')
  plt.plot(testx, sigmoid_arctan(testx,0,0.1), label='w=0.1')
  plt.plot(testx, sigmoid_arctan(testx,0,1.), label='w=1')
  plt.legend()
  plt.title('test sigmoid arctan')

  # test de validité même quand très éloigné du centre de la transition
  plt.figure()
  testx = np.concatenate((-np.logspace(-1,5,1000)[::-1], np.logspace(-1,5,1000)))
  plt.semilogx(np.abs(testx), sigmoid_arctan(testx,0,0.01), label='w=0.01')
  plt.semilogx(np.abs(testx), sigmoid_arctan(testx,0,0.1), label='w=0.1')
  plt.semilogx(np.abs(testx), sigmoid_arctan(testx,0,1.), label='w=1')
  plt.legend()
  plt.title('test sigmoid arctan')

  plt.figure()
  testx = np.linspace(490,510,1000)
  widths = [10, 2., 0.5, 0.25, 1, 1e-1, 1e-2]
  for width in widths:
    plt.plot(testx, sigmoid_arctan(testx,500,width), label='w={}'.format(width))
  plt.legend()
  plt.title('test sigmoid arctan')

  plt.figure()
  testx = np.linspace(1e-5, 1e3)
  widths = [10, 2., 0.5, 0.25, 1, 1e-1, 1e-2]
  pltfun = plt.semilogx
  for width in widths:
  #  pltfun(testx, np.exp(-1.0/sigmoid_arctan(testx,1e-3,width, n=1, p=2))**3, label='w={}'.format(width))
    pltfun(testx, sigmoid_arctan(testx,1e-3,width, n=1, p=2), label='w={}'.format(width))
  pltfun(testx, 1e-10*np.ones_like(testx), color=[0,0,0])
  pltfun(testx, 1e0*np.ones_like(testx), color=[0,0,0])
  plt.legend()
  plt.title('test sigmoid arctan for mass flux')


def fastspy(A, ax, cmap='binary'):
    """"
    Parameters
    ----------
    A : coo matrix
    ax : axis
    """

    m, n = A.shape
    ax.hold(True)

    ax.imshow(A,interpolation='none',cmap=cmap)
    ax.colorbar()
    if 0:
        ax.scatter([i for i in range(np.size(A,1))],
                   [i for i in range(np.size(A,0))],
                   c=A.data, s=20, marker='s',
                   edgecolors='none', clip_on=False,
                   cmap=cmap)

    ax.axis('off')
    ax.axis('tight')
    ax.invert_yaxis()
    ax.hold(False)

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


def mergeDict(prioritary, other, level=0, genealogy='', checkType=True):
    """ Recursively merge two dictionnaries, with precedance for the first one """
    if level==0:  #first call
        out =copy.deepcopy(prioritary)
    else:
        out = prioritary
    for key in other.keys():
        if key not in prioritary.keys():
            out[key] = other[key]
        else:
            #merge
            if isinstance(other[key], collections.Mapping):
                if isinstance(prioritary[key], collections.Mapping):
                    out[key] = mergeDict(prioritary[key], other[key], level=level+1, genealogy='{}.{}'.format(genealogy, key))
                else:
                    raise Exception('Priortary dict has key {}.{} of type {}, whereas it is of type {} in the other one'.format(genealogy, key, type(prioritary[key]), type(other[key])))
            else:
                if checkType:
                  if type(other[key]) == type(prioritary[key]):
                      pass #out[key] = prioritary[key]
                  elif other[key]==None:
                      pass
                  else:
                      raise Exception('Priortary dict has key {}.{} of type {}, whereas it is of type {} in the other one'.format(genealogy, key, type(prioritary[key]), type(other[key])))
    return out

def generateTimeVector(dictInputReference):
    """ Generates the time vector for integration """
    defaults = {'sCase': 'unsteady',
                'unsteady': {'dt':1e-6, 't_f':1e-3},
                'progressive':{
                        'dts': [1e-7,  1e-6, 1e-5], #successive time steps
                        'ntrelax': [ 100,  100], #number of transition time steps after the stabilization steps to transition from on dt to another
                        'nstab': [200,  100,   100], #number of time steps with fixed time steps for each separate dt provided
                        }}
    dictInput = mergeDict(prioritary=dictInputReference, other=defaults)
    if dictInput['sCase']=='unsteady':
            dt=dictInput['unsteady']['dt']
            time = np.arange(0.,dictInput['unsteady']['t_f'],dt)
    elif dictInput['sCase']=='unsteadyProgressif' or dictInput['sCase']=='progressive':
        dts =   dictInput['progressive']['dts'] #successive time steps
        ntrelax = dictInput['progressive']['ntrelax'] #number of transition time steps after the stabilization steps to transition from on dt to another
        nstab =  dictInput['progressive']['nstab'] #number of time steps with fixed time steps for each separate dt provided

        time = [0.]
        for i in range(len(dts)-1):
            for j in range(nstab[i]):
                time.append(time[-1] + dts[i])
            for j in range(ntrelax[i]):
                time.append(time[-1]+ (ntrelax[i]-j)/ntrelax[i]*dts[i] +  j/ntrelax[i]*dts[i+1])
        for j in range(nstab[-1]):
            time.append(time[-1] + dts[-1])
        time = np.array(time)
    else:
        raise Exception('unknown time stepping configuration "{}"'.format(dictInput['sCase']))

    if 'globalScaling' in dictInputReference.keys(): # global scaling to easily reduce time step sizes
        time = time*dictInputReference['globalScaling']
    return time

#raise Exception('attention  tu dois finir cette implémentation')
def interpExtrap1D(x, y, kind='linear'):
    """ Interpolateur qui extrapole avec des valeurs constantes, évite les problèmes posés par scipy.interp1d """
    import scipy.interpolate
#    from scipy import array
    def extrap1d(interpolator):
        xs = interpolator.x
        ys = interpolator.y

#        def pointwise(x):
#            if x < xs[0]:
#                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
#            elif x > xs[-1]:
#                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
#            else:
#                return interpolator(x)

        def ufunclike(xnew):
            Iinterp = np.intersect1d( np.where(xnew>xs[0])[0], np.where(xnew<xs[-1])[0] ).astype(int)
            Iextrap_low = np.where(xnew<=xs[0])
            Iextrap_up  = np.where(xnew>=xs[-1])
            ynew=np.zeros_like(xnew)
            ynew[Iinterp] = interpolator(xnew[Iinterp])
            ynew[Iextrap_low] = ys[0]
            ynew[Iextrap_up]  = ys[-1]
            return ynew
#            return np.array(map(pointwise, np.array(xnew)))

        return ufunclike
    return extrap1d(scipy.interpolate.interp1d(x,y,#axis=0,
                                               kind=kind))

if __name__=='__main__':
    import matplotlib.pyplot as plt
    sDict = {'sCase': 'unsteady',
            'unsteady': {'dt':1e-6, 't_f':1e-3},
            'progressive':{
                    'dts': [1e-7,  1e-6, 1e-5], #successive time steps
                    'ntrelax': [ 100,  100], #number of transition time steps after the stabilization steps to transition from on dt to another
                    'nstab': [200,  100,   100], #number of time steps with fixed time steps for each separate dt provided
                    }}
    time = generateTimeVector(sDict)
    plt.figure()
    plt.plot(time)
    plt.title('time vector')
    plt.xlabel('index')
    plt.ylabel('time')

    plt.figure()
    plt.plot(np.diff(time))
    plt.title('time vector gradient')
    plt.xlabel('index')
    plt.ylabel('dt')

    # test merge
    dict1 = {'a': 1.,
             'b':{'c':2,
                  'd':4,
                  }}
    dict2 = {'a': 1.3,
             'b':{'c':23,
                  'f': 5,
                  'e':{'test':'a word',}
                  }}

    dict3 = mergeDict(dict1, dict2)
    print(dict3)


    # test de l'interpolation avev extrapolation linéaire
    x= np.array([0., 1., 2.])
    y = np.array([0., 2., 4.])

    xnew = np.array([-1, 0., 0.5, 1.5, 2., 4.])
    ynew = interpExtrap1D(x,y,kind='linear')(xnew)
    ynew2 = scipy.interpolate.interp1d(x,y,kind='linear', fill_value='extrapolate')(xnew)
    plt.figure()
    plt.plot(x,y,label='original', marker='+', color='b')
    plt.scatter(xnew,ynew,  label='new custom', marker='o', color='r')
    plt.scatter(xnew,ynew2, label='interp1d', marker='x', color='g')
    plt.legend()
    plt.title('Validation de mon interpolation avec extrapolation constante')