import numpy as np

def fix_Mpositive(xy):
  M = np.sum(xy,axis=1)
  neg_idc = np.where(M<0)
  xy[neg_idc] = - xy[neg_idc]
  return xy

def fix_Z2symmetry(xy):
  Ns,N = xy.shape
  xy = fix_Mpositive(xy)
  xy[:Ns//2,:] = - xy[:Ns//2,:]
  return xy