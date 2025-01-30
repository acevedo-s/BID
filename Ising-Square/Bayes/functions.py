import numpy as np
from scipy.special import gammaln

def get_probs(ID,remp):
  logprobs = (gammaln(ID + np.double(1))
  - gammaln(remp + 1)
  - gammaln(ID - remp + 1)
  - ID * np.log(np.double(2)))
  probs = np.exp(logprobs)
  return probs / np.sum(probs)

def get_ID(d,remp):
  return d[0] + d[1] * remp

def run_MCMC(N_iter,
             d_0,
             delta,
             remp,
             Cemp):
  dynamics = np.zeros(shape=(N_iter,2))
  logRs = np.zeros(shape=(N_iter,))
  fluctuations_counter = 0

  d = d_0
  for i in range(N_iter):
    d_p =  d + np.array(d) * np.random.uniform(low=-delta,high=delta,size=2) # proposal
    ID = get_ID(d,remp)
    ID_p = get_ID(d_p,remp)
    logR = np.sum(Cemp * np.log((get_probs(ID_p,remp) / get_probs(ID,remp))))
    logRs[i] = logR
    if logR > np.log(1):
      d = d_p
    else:
      u = np.random.uniform()
      if np.log(u) < logR:
        d = d_p
        fluctuations_counter += 1
    dynamics[i] = d
  print(f'final d: {d}')
  return dynamics,logRs,fluctuations_counter