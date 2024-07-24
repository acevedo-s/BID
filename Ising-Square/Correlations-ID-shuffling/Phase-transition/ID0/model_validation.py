import numpy as np
import sys,os
import matplotlib.pyplot as plt
from dadapy import Hamming
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

eps = 1E-7
figsfolder = f'results/figs/'
optfolder = f'results/opt/'
histfolder = f'../distances/results/hist/'
crossed_distances = 0

alphamin = float(sys.argv[1])

figh,axh = plt.subplots(1,)
T_list = np.arange(2,2.1+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.2,2.4+eps,.02))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.5,3+eps,.1)))
N_list = [4096]

print(T_list)
alphamax_list = np.arange(.1,.8+eps,.1) # orders of quantile for P(r)
Ns = 10000
for N_id,N in enumerate(N_list):
  k = int(round(np.sqrt(N)))
  for T_id,T in enumerate(T_list):
    ### optimization
    rmaxs = np.empty(shape=(len(alphamax_list)))
    sigmas = np.empty(shape=rmaxs.shape)
    alphas = np.empty(shape=rmaxs.shape)
    logKLs = np.empty(shape=rmaxs.shape)

    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
    T=T,
    L=N,
    Ns=Ns,
    # t=k,
    resultsfolder=histfolder,
    )
    for alphamax_id,alphamax in enumerate(alphamax_list):
      outputfile = optfolder + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
      A = np.loadtxt(outputfile,delimiter=',')
      rmaxs[alphamax_id]  = A[0]
      sigmas[alphamax_id]  = A[1]
      alphas[alphamax_id]  = A[2]
      logKLs[alphamax_id]  = A[3]
    minKL_id = np.where(np.isclose(logKLs,np.nanmin(logKLs)))[0][0]
    if rmaxs[minKL_id] == 1:
      minKL_id += 1
    sigma = sigmas[minKL_id]
    alpha = alphas[minKL_id]
    rmax = rmaxs[minKL_id]

    ### id min
    H.set_r_quantile(alphamin)
    rmin = H.r
    idmin = H.r_idx
    H.r = None
    H.r_idx = None

    ### id max
    idmax = np.where(H.D_values==rmax)[0][0]
    remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
    Pemp = H.D_probs[idmin:idmax+1]
    Pemp /= np.sum(Pemp)
    Pmodel = jnp.zeros(shape=remp.shape, dtype=jnp.float64)
    Op = Optimizer(
                  remp=remp,
                  sigma_r=jnp.double(sigma),
                  alpha_r=jnp.double(alpha),
                  Pmodel=Pmodel,
                  )
    Op = compute_Pmodel(Op)
    axh.plot(remp/N,
            Pemp,
            'x',
              label=f'{T=:.2f}',
            )
    axh.plot(remp/N,
              Op.Pmodel,
              color='black')
      
axh.set_yscale('log')
box = axh.get_position()
axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axh.grid(True)
axh.set_xlabel('r/N')
axh.set_ylabel('P(r)')
axh.set_title(f'{alphamin=:.5f}')
figh.savefig(figsfolder + 'model_validation.png',
             bbox_inches='tight',
             )
