from dadapy.base import Base
import multiprocessing
import numpy as np
import os
from dadapy._utils.utils import compute_cross_nn_distances
from jax import numpy as jnp, jit,lax
from jax import devices as jdevices
# from jax.config import config
from jax.experimental.host_callback import call
from scipy.special import gammaln



cores = multiprocessing.cpu_count()


class Hamming(Base):
  
  def __init__(
              self,
              q=2,
              coordinates=None,
              distances=None,
              crossed_distances=0,
              maxk=None,
              verbose=False,
              njobs=cores,
              ):
    super().__init__(
          coordinates=coordinates,
          distances=distances,
          maxk=maxk,
          verbose=verbose,
          njobs=njobs,
      )
    self.q = q
    self.coordinates = coordinates
    self.metric='hamming'
    self.crossed_distances=crossed_distances
    # self.maxk=maxk

    self.r = None
    self.r_idx = None
    self.a = None
    self.a_idx = None
    self.b = None
    self.b_idx = None
    self.nA = None
    self.nB = None
    self.cA = None
    self.cB = None
    self.D_values = None
    self.D_counts = None
    self.D_probs = None
    self.D_probs_local = None
    self.D_values_local = None
    self.D_mu_emp = None
    self.D_var_emp = None
    
    assert (self.crossed_distances is not None), "define crossed_distances to be 0 or 1"

  def compute_distances(self,
                        sort=False,
                        check_format=True,
                        jcompute=True,
                        ):
    assert (self.crossed_distances==0)
    if self.q == 2 and jcompute:
      self.distances = jcompute_distances(X1=self.coordinates,
                                          X2=self.coordinates,
                                          crossed_distances=self.crossed_distances,
                                          check_format=check_format,
                                          sort=sort,
                                          )    
  """ TODO: MODIFY HISTOGRAM ROUTINE TO DISCARD THE TRIVIAL ZEROS 
  WHEN CROSSED_DISTANCES = 1  """ 
  
  def remove_D_outliers(self):
    # left outliers:
    aux_idx = int(len(self.D_values)/2) 
    id_cut = np.asarray(np.diff(self.D_values[:aux_idx])>1).nonzero()[0]
    if len(id_cut)>=1:
      id_cut = id_cut[-1]
      id_cut += 1
    else:
      id_cut = 0
    self.D_values = self.D_values[id_cut:]
    self.D_counts = self.D_counts[id_cut:]
    self.D_probs = self.D_probs[id_cut:]   
    # # right outliers:
    aux_idx = int(len(self.D_values)/2)
    id_cut = np.asarray(np.diff(self.D_values)>1).nonzero()[0]
    if len(id_cut)>=1:
      id_cut = id_cut[0]
    else:
      id_cut = None
    self.D_values = self.D_values[:id_cut]
    self.D_counts = self.D_counts[:id_cut]
    self.D_probs = self.D_probs[:id_cut]  

  def D_histogram(self,
                  compute_flag=0,
                  save=False,
                  T=None,
                  precision_T=2,
                  digits_T=1,
                  L=None,
                  Ns=None,
                  k=None,
                  t=None,
                  r_id=None,
                  resultsfolder = f'results/hist/',
                  ):
    
    if save:
      os.makedirs(resultsfolder,exist_ok=True)
    if L is not None:
      resultsfolder += f'L{L}'
    if T is not None:
      resultsfolder += f'_T{T:{digits_T}.{precision_T}f}'
    if k is not None:
      resultsfolder += f'_k{k}'
    if Ns is not None:
      resultsfolder += f'_Ns{Ns}'
    if t is not None:
      resultsfolder += f'_t{t}'
    if r_id is not None:
      resultsfolder += f'_rid{r_id}'
    if self.crossed_distances==1:
      resultsfolder += f'_crossed'
    c_fname = resultsfolder + f'D_counts'

    if compute_flag:
      self.D_values, self.D_counts = np.unique(self.distances,
                                              return_counts=True)

      if self.crossed_distances==0:
        Nsamples = self.distances.shape[0]
        assert self.D_values[0] == 0 # trivial zeros
        Nzeros = int(Nsamples * (Nsamples+1) / 2) # trivial zeros, Gauss sum of them
        self.D_counts[0] -= Nzeros
        if self.D_counts[0] == 0:
          self.D_values = self.D_values[1:]
          self.D_counts = self.D_counts[1:]
      self.D_probs = self.D_counts / np.sum(self.D_counts)

      if save:
        np.savetxt(fname=c_fname+'.txt',
                   X=np.transpose([self.D_values,self.D_counts]),
                   fmt="%d,%d"
                   )
    else:
      f = np.loadtxt(c_fname+'.txt',delimiter=',',dtype=int)
      self.D_values = f[:,0]
      self.D_counts = f[:,1]
      self.D_probs = self.D_counts / np.sum(self.D_counts)

      
  def set_r_quantile(self,
                    alpha,
                    round=True,
                    precision=10
                    ):
    
    if round:
      alpha = np.round(alpha,precision)
      self.D_probs = np.round(self.D_probs,precision)

    indices = np.where(np.cumsum(self.D_probs) <= alpha)[0]
    if (len(indices)==0):
      self.r_idx = 0
    else:
      self.r_idx = indices[-1]

    self.r = int(self.D_values[self.r_idx])
    return

  def set_b_quantile(self,
                    alpha,
                    round=True,
                    precision=10
                    ):
    
    if round:
      alpha = np.round(alpha,precision)
      self.D_probs = np.round(self.D_probs,precision)

    indices = np.where(np.cumsum(self.D_probs) <= alpha)[0]
    self.b_idx = indices[-1]
    self.b = int(self.D_values[self.b_idx])
  
  def set_r(self,r=None,n_sigma=3):
    if r is not None:
      self.r = r
    else:
      self.compute_moments()
      r = int(np.round(self.D_mu_emp-n_sigma*np.sqrt(self.D_var_emp),0))
      if r >= self.D_values[0]:
        self.r = r
      else:
        self.r = self.D_values[0]

  def set_r_idx(self,r_idx=None):
    if r_idx is not None:
      self.r_idx = r_idx
    else:
      aux = np.asarray(self.r==self.D_values).nonzero()[0]
      if len(aux)>0:
        self.r_idx = aux[0]
      else:
        self.r_idx = 0
  
  def set_a_b(self,a,b):
    self.a = a
    self.a_idx = np.where(self.D_values==self.r+self.a)[0][0]
    self.b = b
    self.b_idx = np.where(self.D_values==self.r+self.b)[0][0]
    self.D_values_local = self.D_values[self.r_idx:self.b_idx+1]
    self.D_probs_local = self.D_probs[self.r_idx:self.b_idx+1]
    
  
  def count(self,):
    n0 = (self.distances < self.r).sum(axis=1)
    self.nA = (self.distances <= self.r + self.a).sum(axis=1) - n0
    self.nB = (self.distances <= self.r + self.b).sum(axis=1) - n0
  
  def count_fast(self):
    assert self.a_idx is not None
    assert self.b_idx is not None
    self.cA = self.D_counts[self.r_idx:self.a_idx+1].sum() 
    self.cB = self.D_counts[self.r_idx:self.b_idx+1].sum()

  def compute_moments(self):
    self.D_mu_emp = np.dot(self.D_probs,self.D_values)
    self.D_var_emp = np.dot(self.D_probs,self.D_values**2) - self.D_mu_emp**2

  def logprob_free(self,d,r):
    return (gammaln(d+1)
      -gammaln(d-r+1)
      -gammaln(r+1)
      -d*np.log(self.q)
      +r*np.log(self.q-1)
      )

  def logprobs_free(self,d,rs):
    return np.array([self.logprob_free(d,r) for r in rs])

def check_data_format(X):

  U = np.unique(X)
  e1,e2 = np.unique(X)
  assert(e1 == -1 and e2 == 1), f'spins have to be formatted to -+1, but {np.unique(X)=}'

def jcompute_distances(X1,
                       X2,
                       crossed_distances,
                       check_format=True,
                       sort=False,
                       ):
  """ This routine works for Ising spins variables defined as +-1 """
  # config.update('jax_platform_name', 'cpu')
  jdevices("cpu")[0]
  X1 = jnp.array(X1).astype(jnp.int32)
  X2 = jnp.array(X2).astype(jnp.int32)

  if check_format:
    check_data_format(X1)
    check_data_format(X2)

  Ns1,N = X1.shape
  Ns2 = X2.shape[0]

  distances = jnp.zeros(shape=(Ns1,Ns2),dtype=jnp.int32)
  sample_idx = 0
  lower_idx = 0
  pytree = {"crossed_distances":crossed_distances,
            "D":distances,
            "X1":X1,
            "X2":X2,
            "sample_idx":sample_idx,
            "lower_idx":lower_idx,
            "Ns1":Ns1,
            "Ns2":Ns2,
            "N":N,
            }

  pytree = lax.fori_loop(lower=0,
                upper=Ns1,
                body_fun=_jcompute_distances,
                init_val=pytree
                )
  
  if sort:
    return np.array(jnp.sort(pytree["D"]))
  else:
    return np.array(pytree["D"])

@jit
def _jcompute_distances(sample_idx,pytree):
  pytree["sample_idx"] = sample_idx
  # call(lambda x: print(f'idx={x}'),pytree["idx"])

  pytree = lax.cond(pytree["crossed_distances"],
                    _set_lower_idx_true,
                    _set_lower_idx_false,
                    pytree)
  pytree = lax.fori_loop(lower=pytree["lower_idx"],
                        upper=pytree["Ns2"],
                        body_fun=compute_row_distances,
                        init_val=pytree
                        )
  return pytree

def _set_lower_idx_true(pytree):
  pytree["lower_idx"] = 0
  return pytree
def _set_lower_idx_false(pytree):
  pytree["lower_idx"] = pytree["sample_idx"] + 1 
  return pytree

@jit
def compute_row_distances(_idx,pytree):
  pytree["D"] = pytree["D"].at[pytree["sample_idx"],_idx].set(
    jnp.int32((pytree["N"]-
      jnp.dot(
        pytree["X1"][pytree["sample_idx"],:],
        pytree["X2"][_idx,:]
      )
      )/2)
                )
  return pytree