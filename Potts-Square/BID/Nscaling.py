import sys,os
import matplotlib.pyplot as plt
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import BID

rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)
markers = ['p','o','h','^','s']

plot_id = 0


figsfolder = f'results/figs/'
_optfolder = f'results/opt/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7

fig,ax = plt.subplots(1)

L_list = np.array([40,50,60,70,80,90,100])
N_list = 3*L_list**2
T_list = np.flip(np.array([.6,.7,.748,.8,.9]))

alphamin = 1E-3 # order of quantile for P(r)
alphamax = .1 #5E-3
Nsteps = int(1E6)
seed = 1
delta = 5E-4

sigmas = np.empty(shape=(len(L_list),
                         len(T_list)
                        )
                  )
rmaxs = np.empty(shape=sigmas.shape,dtype=int)
alphas = np.empty(shape=sigmas.shape)
logKLs = np.empty(shape=sigmas.shape)

for T_id,T in enumerate(T_list):
  for L_id,L in enumerate(L_list):
    optfolder0 = _optfolder + f'L{L}/T{T:.3f}/'
    B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder0,
            )
    (rmaxs[L_id,T_id],
    sigmas[L_id,T_id],
    alphas[L_id,T_id],
    logKLs[L_id,T_id]) = B.load_results()

    

for T_id,T in enumerate(T_list):
  ax.scatter(N_list,
            sigmas[:,T_id]/N_list,
            color=colors[plot_id],
            edgecolors='black',
            label=f'{T=:.2f}',
            marker=markers[plot_id],
            s=80,
            )
  ax.plot(N_list,
          sigmas[:,T_id]/N_list,
          zorder=0,
          color=colors[plot_id],
          )
  plot_id += 1
ax.set_xlabel(r'$N$')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(r'BID/N')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='center right',
          prop={'size':18},
          ncol=2)
ax.set_title('Potts square')


fig.savefig(figsfolder + f'Potts-Nscaling.pdf',
            bbox_inches='tight',
            )      