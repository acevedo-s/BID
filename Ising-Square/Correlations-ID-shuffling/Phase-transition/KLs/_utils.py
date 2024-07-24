import numpy as np

def set_common_support(rs,probs,rmin,rmax,normalize=True):
  idmin = np.where(rs==rmin)[0][0]
  idmax = np.where(rs==rmax)[0][0]
  rs = rs[idmin:idmax+1]
  probs = probs[idmin:idmax+1]
  if normalize:
    probs /= np.sum(probs)
  return rs, probs

def KLd_PQ(P,Q):
  return np.dot(P,(np.log(P)-np.log(Q)))

def regularize_hists(N,rs_emp,probs):

  rs = np.arange(N+1,dtype=int)
  reg_probs = np.ones(shape=(N+1,),dtype=float) / 10**9
  reg_probs[rs_emp] = probs
  return rs,reg_probs