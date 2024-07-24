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
np.set_printoptions(precision=3)
markers = ['p','o','h','^','s']
plot_id = 0
Ts = .745

log_scale = 1
figsfolder = f'results/figs/'
optfolder0 = f'results/opt/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
crossed_distances = 0
precision_T = 3
Ns = 5000 

T_list = np.arange(.5,.7+eps,0.1)
deltaT = .004
T_list = np.concatenate((T_list,
                         np.arange(.71,.738+eps,deltaT))
                         )
T_list = np.concatenate((T_list,
                         np.arange(.748,.77+eps,deltaT))
                         )
T_list = np.concatenate((T_list,
                         np.arange(.8,1+eps,.1))
                         )
print(f'{T_list=}')

fig,ax = plt.subplots(1)
figa,axa = plt.subplots(1)
figKL,axKL = plt.subplots(1)

L_list = np.array([60,80,100])
N_list = 3*L_list**2

alphamin = 1E-3 # order of quantile for P(r)
alphamax = .1 #5E-3
Nsteps = int(1E6)
delta = 5E-4
seed = 1

sigmas = np.empty(shape=(len(L_list),
                         len(T_list)
                        )
                  )
rmaxs = np.empty(shape=sigmas.shape,dtype=int)
alphas = np.empty(shape=sigmas.shape)
logKLs = np.empty(shape=sigmas.shape)

for T_id,T in enumerate(T_list):
  for L_id,L in enumerate(L_list):
    optfolder = optfolder0 + f'L{L}/T{T:.3f}/'
    B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder,
            )
    (rmaxs[L_id,T_id],
    sigmas[L_id,T_id],
    alphas[L_id,T_id],
    logKLs[L_id,T_id]) = B.load_results()
    if logKLs[L_id,T_id] == np.inf:
      sigmas[L_id,T_id] = None
      print(f'{L=},{T=:.3f} failed')
# sys.exit()
for L_id,L in enumerate(L_list):
  N = 3 * L**2
  ax.scatter(T_list,
            sigmas[L_id,:]/N,
            color=colors[plot_id],
            edgecolors='black',
            marker=markers[plot_id],
            label=f'{L=}',
            )
  ax.plot(T_list,
          sigmas[L_id,:]/N,
          zorder=0,
          color=colors[plot_id],
          )
  axa.plot(T_list,
          alphas[L_id,:],
          zorder=0,
          color=colors[plot_id],
          label=f'{L=}',
          )
  axKL.plot(T_list,
          logKLs[L_id,:],
          zorder=0,
          color=colors[plot_id],
          label=f'{L=}',
          )
  plot_id += 1
ax.set_xlabel(r'$T$')
if log_scale:
        ax.set_yscale('log')
ax.set_ylabel(r'BID/N')
ax.vlines(Ts,
          ax.get_ylim()[0],
          ax.get_ylim()[1],
          color='black',
          linestyles='dashed',
          zorder=0)
axa.vlines(Ts,
          axa.get_ylim()[0],
          axa.get_ylim()[1],
          color='black',
          linestyles='dashed',
          zorder=0)
axKL.vlines(Ts,
          axKL.get_ylim()[0],
          axKL.get_ylim()[1],
          color='black',
          linestyles='dashed',
          zorder=0)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.grid()
ax.legend(loc='center right')
fig.savefig(figsfolder + f'Potts-Td0.pdf',
            bbox_inches='tight',
            )  

box = axa.get_position()
axa.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axa.legend(loc='center left', bbox_to_anchor=(1, 0.5))

figa.savefig(figsfolder + f'Td1.pdf',
            bbox_inches='tight',
            ) 
  
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))

figKL.savefig(figsfolder + f'TKL.pdf',
            bbox_inches='tight',
            )      