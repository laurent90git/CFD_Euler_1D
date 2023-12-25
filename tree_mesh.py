# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:47:50 2023

Simple 1D tree structure for adaptive mesh refinment

@author: lfrancoi
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy

BDEBUG = False
MAX_LEVEL = 14
class Node():
  def __init__(self, center, dx, value, level=0, father=None,
               child_id=0, min_level=0, max_level=MAX_LEVEL):
    self.center = center
    self.dx = dx
    self.xf = np.array([center-dx/2, center+dx/2])
    self.children = []
    self.value = value # scalar or vector data stored in this cell
    self.level = level
    self.father= father
    self.is_destroyed = False
    self.child_id = child_id
    self.child_deletion_agreement = [False, False]
    self.max_level = max_level
    self.min_level = min_level
    if father is None:
      assert level==0, 'only 0-th level nodes may have no father'
    
  def hasChildren(self):
    return len(self.children)>0
    
  def destroy(self):
    if self.level==self.min_level:
      return # we cannot delete level-0 nodes
    # if not self.is_destroyed:
    self.father.require_lineage_destruction(self.child_id)
      # self.is_destroyed = True
      
  def __del__(self):
    # print('being destroyed')
    pass
    # assert self.is_destroyed
    # assert not self.father.hasChildren()
    
  def reset_lineage_destruction(self):
    """ Reset destruction counter, i.e. if not all children have been asked to be
    destroyed, we keep them and reset the requirement for destruction """
    self.child_deletion_agreement = [False, False]
    
  
  def recursive_destruction_reset(self):
    assert not self.is_destroyed
    self.reset_lineage_destruction()
    for c in self.children:
      c.recursive_destruction_reset()
    
  def require_lineage_destruction(self, child_id):
    """ Destroy children leafs and project values conservatively """
    if not self.hasChildren():
      raise Exception('weird')
    self.child_deletion_agreement[child_id] = True
    
    if all(self.child_deletion_agreement):
      # both children have asked for, we can destroy them
      # we recover their values first
      self.value = 0.5*( self.children[0].value + self.children[1].value )
      for c in self.children:
        assert not c.is_destroyed
        c.is_destroyed = True
      self.children = []
    
  def refine(self):
    """ Creates children """
    # if self.center== -9.6875:
      # print('here')
    if self.hasChildren():
      raise Exception('node is already refined')
      
    if self.level==self.max_level:
      # raise Exception('maximum refinment level reached')
      return
      
    # TODO: better interpolation
    self.children = [Node(center=self.center-self.dx/4, dx=self.dx/2, value=self.value,
                          level=self.level+1, father=self, child_id=0,
                          min_level=self.min_level, max_level=self.max_level),
                     Node(center=self.center+self.dx/4, dx=self.dx/2, value=self.value,
                          level=self.level+1, father=self, child_id=1,
                          min_level=self.min_level, max_level=self.max_level)]
    
  def recursive_refine(self, target_level):
    if self.level>=target_level:
      return
    if not self.hasChildren():
      self.refine()
    for c in self.children:
      c.recursive_refine(target_level)
      
  def getMesh(self):
    """ Recursive function to gather the outer leafs and construct the mesh """
    if not self.hasChildren():
      return self.xf, self.center, self.level, self.value, self
    else:
      left_xfs,  left_centers,  left_levels,  left_values,  left_nodes  = self.children[0].getMesh()
      right_xfs, right_centers, right_levels, right_values, right_nodes = self.children[1].getMesh()
      faces = np.hstack((left_xfs[:-1], right_xfs)) # remove duplicate face between children
      if BDEBUG:
        assert np.unique(faces).size == faces.size
      return faces, \
             np.hstack((left_centers, right_centers)), \
             np.hstack((left_levels, right_levels)), \
             np.vstack((left_values, right_values)), \
             np.hstack((left_nodes, right_nodes))
             
    
  def getLeafs(self):
    """ Returns the leaf nodes of its lineage """
    if not self.hasChildren():
      return [self]
    else:
      return [c.getLeafs() for c in self.children]
    
  def getMaxSubLevel(self):
    """ Returns the maximum level among its lineage (useful for grading) """
    if not self.hasChildren():
      return self.level
    else:
      return max([c.getMaxSubLevel() for c in self.children])
    
  def getMaxLeftSublevel(self):
    """ Returns the maximum level among its lineage
    at the left boundary of its domain (useful for grading) """
    if not self.hasChildren():
      return self.level
    else:
      return self.children[0].getMaxLeftSublevel()
    
  def getMaxRightSublevel(self):
    """ Returns the maximum level among its lineage
    at the left boundary of its domain (useful for grading) """
    if not self.hasChildren():
      return self.level
    else:
      return self.children[1].getMaxRightSublevel()
      
  def gradeTree(self):
    if not self.hasChildren():
      return
    
    max_left_level  =  self.children[0].getMaxRightSublevel()
    max_right_level =  self.children[1].getMaxLeftSublevel()
    if max_left_level < max_right_level-1: # refine left child
      self.children[0].recursiveRightRefine(max_right_level-1)
    elif max_right_level < max_left_level-1: # refine right child
      self.children[1].recursiveLeftRefine(max_left_level-1)
      
    for c in self.children:
      c.gradeTree()
      
      
  def recursiveRightRefine(self, target_level):
    if self.level>=target_level:
      # raise Exception('why refine ?')
      return
    if not self.hasChildren():
      self.refine()
    self.children[1].recursiveRightRefine(target_level)
    
  def recursiveLeftRefine(self, target_level):
    if self.level>=target_level:
      # raise Exception('why refine ?')
      return
    if not self.hasChildren():
      self.refine()
    self.children[0].recursiveLeftRefine( target_level)
      
  def recursive_set_value(self,fun):
    self.value = fun(self.center)
    if self.hasChildren():
      for c in self.children:
        c.recursive_set_value(fun)
    
#%%
if __name__=='__main__':
  node1 = Node(center=0, dx=1, value=np.array([1.,2.]))
  # node1.refine()
  
  # node1.children[0].refine()
  # node1.children[0].children[0].refine()
  # node1.children[0].children[1].refine()
  # node1.recursiveLeftRefine(6)
  # node1.recursiveRightRefine(10)
  
  # node1.children[1].recursiveLeftRefine(6)
  # node1.children[0].recursiveLeftRefine(8)
  # node1.getMaxSubLevel()
  # node1.gradeTree()
  
  # problème de grading vers les BCs
  node1.recursive_refine(target_level=8)
  node = node1
  for i in range(2):
    ii = np.round( np.random.rand(1) ).astype(int)[0]
    # ii=1
    node = node.children[ii]
    
  node.children[0].destroy()
  node.children[1].destroy()
  
  #%%
  node1.gradeTree()
  
  node1.recursive_set_value(fun=lambda x: np.array([x,10+x,100+x]))

  # node1.recursive_refine(4)
  faces, centers, levels, values, nodes = node1.getMesh()
  
  plt.figure()
  for i in range(centers.size):
    plt.plot(faces[i:i+2], [levels[i], levels[i]], color='tab:blue')
    plt.plot(centers[i], levels[i], color='tab:blue', marker='x')
  plt.xlabel('x')
  plt.ylim(0, 10)
  plt.grid()
  plt.ylabel('level')
  