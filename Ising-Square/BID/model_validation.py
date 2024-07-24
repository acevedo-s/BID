import sys,os
import matplotlib.pyplot as plt
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
np.set_printoptions(precision=3)
markers = ['s']

figsfolder = f'results/figs/model_validation/'
optfolder0 = f'results/opt/'
histfolder = f'../distances/results/hist/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
crossed_distances = 0
precision_T = 2
Ns = 5000 

metric = 'hamming'
crossed_distances = 0

figh,axh = plt.subplots(1)
alphamin = 0
alphamax = .05
Nsteps = int(1E6)
delta = 5E-4
seed = 1

T_list = [1.7,2.3,4]
L_list = [100]
for T_id,T in enumerate(T_list):
  for L_id,L in enumerate(L_list):
    optfolder = optfolder0 + f'L{L}/T{T:.2f}/'
    B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder,
            )
    remp,Pemp,Pmodel = B.load_fit()
    rmax,sigma,alpha,logKL = B.load_results()
    
    N = L**2
    axh.plot(remp/N,
              Pemp,
              'x',
              label=f'{T=:.2f}',
              zorder=0,
              )
    if logKL < np.inf:
      axh.plot(remp/N,
            Pmodel,
            #  label=r'$P_{model}$',
            zorder=1,
            color='black',
            )

axh.set_xlabel(r'$r/N$')
axh.set_yscale('log')
axh.set_ylabel(r'P(r)')

# box = axh.get_position()
# axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axh.legend()

figh.savefig(figsfolder + f'Square-model_validation.pdf',
             bbox_inches='tight',
             )      