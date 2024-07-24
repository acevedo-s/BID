
import matplotlib.pyplot as plt
from dadapy import Hamming
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *

#for fancy plotting
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=7)
# markers = ['p','p','o','o','x','x']
markers = ['s']

resultsfolder = f'results/opt/'
os.makedirs(resultsfolder,exist_ok=True)
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)

eps = 1E-7
crossed_distances = 0


T_list = np.arange(2,2.1+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.2,2.4+eps,.01))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.5,3.+eps,.1)
                         )
                         )
T_list = np.flip(T_list)
N_list = [4096]

alphamin = float(sys.argv[1]) # order of quantile for P(r)
print(f'{alphamin=}')
# alphamax_list = [.1,.2,.3,.4,.5,.6,.7]
# alphamax_list = np.arange(.1,1.+eps,.2)
alphamax_list = np.arange(.1,.4+eps,.1) # orders of quantile for P(r)
# alphamax_list = [1]
delta0 = 5E-2
print(f'{delta0=}')
delta1 = 1E-2
print(f'{delta1=}')
Nsteps = int(5E5)
print(f'{Nsteps=}')
seed = 111
print(f'{seed=}')
histfolder = f'../distances/results/hist/'
delete_previous_file = 1

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
alphamax_id = task_id
alphamax = alphamax_list[alphamax_id]
print(f'{alphamax=:.5f}')

for N_id,N in enumerate(N_list):
  L = int(np.round(np.sqrt(N)))
  k_list = [L]
  for k_id, k in enumerate(k_list):
    delta = delta0
    alpha0 = 1. # slope of ID 
    print(f'{alpha0=}')
    for T_id,T in enumerate(T_list):
      print(f'{T=:.2f}')
      outputfile = resultsfolder + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
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
      if T_id == 0:
        sigma0 = H.D_mu_emp
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
      ### annealing of parameters
      alpha0 = Op.alpha
      sigma0 = Op.sigma
      delta = delta1