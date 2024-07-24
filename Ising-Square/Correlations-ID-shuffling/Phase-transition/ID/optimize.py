import matplotlib.pyplot as plt
from dadapy import Hamming
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *
from _utils_initial_condition import initial_condition

eps = 1E-7
crossed_distances = 0
N_list = [4096]

try:
  task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
except:
  task_id = 1
alphamin = float(sys.argv[1]) # order of quantile for P(r)
print(f'{alphamin=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
delta0 = 1E-3
print(f'{delta0=}')
Nsteps = int(5E5)
print(f'{Nsteps=}')
seed = 111
print(f'{seed=}')
histfolder = f'../distances/results/shuff/hist/'
delete_previous_file = 1

alphamax =  float(sys.argv[3])
print(f'{alphamax=:.5f}')

for N_id,N in enumerate(N_list):
  L = int(np.round(np.sqrt(N)))
  k_list = [L]
  for k_id, k in enumerate(k_list):
    optfolder = f'results/opt/L{N}/k{k}/T{T:.2f}/alphamin{alphamin:.5f}/alphamax{alphamax:.5f}/'
    os.makedirs(optfolder,exist_ok=True)
    delta = delta0
    rmax0,sigma0,alpha0,logKL0 = initial_condition(N,k,T,alphamin,alphamax)
    print(f'{alpha0=}')
    print(f'{T=:.2f}')
    outputfile = optfolder + f'rid{task_id}.txt'
    if delete_previous_file:
      os.system(f'rm -f {outputfile}')
    if T<1:
      Ns = 5000
    else: 
      Ns = 10000
    H = Hamming(
                crossed_distances=crossed_distances,
                )
    H.D_histogram(
                  T=T,
                  L=N,
                  # t=k,
                  Ns=Ns,
                  r_id=task_id,
                  resultsfolder=histfolder,
                  )
    H.compute_moments()    
    ### id MIN
    H.set_r_quantile(alphamin)
    rmin = H.r
    idmin = H.r_idx
    H.r = None
    H.r_idx = None
    ### id MAX
    H.set_r_quantile(alphamax)
    rmax = H.r
    idmax = H.r_idx
    H.r = None
    H.r_idx = None
    remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
    Pemp = jnp.array(H.D_probs[idmin:idmax+1], dtype=jnp.float64)
    Pemp /= jnp.sum(Pemp)
    Pmodel = jnp.zeros(shape=Pemp.shape, dtype=jnp.float64)
    key0 = random.PRNGKey(seed)
    Op = Optimizer(key=key0,
                  sigma=jnp.double(sigma0),
                  alpha=jnp.double(alpha0),
                  delta=jnp.double(delta),
                  remp=remp,
                  Pemp=Pemp,
                  Pmodel=Pmodel,
                  Nsteps=Nsteps,
                  )
    Op = minimize_KL(Op)
    print(f'{Op.sigma=:.3f},{Op.alpha:3f},{jnp.log(Op.KL)=:.2}')
    print(f'{rmax:.3f},{Op.sigma:.8f},{Op.alpha:8f},{np.log(Op.KL):.8f}',
            file=open(outputfile, 'a'))