import sys,os
import matplotlib.pyplot as plt
import numpy as np
os.environ["JAX_ENABLE_X64"] = "True"
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

log_scale = 0
figsfolder = f'results/figs/'
optfolder0 = f'results/opt/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
crossed_distances = 0
precision_T = 3
Ns = 5000 

T_list = np.loadtxt('../distances/T_list40.txt')

fig,ax = plt.subplots(1)
figa,axa = plt.subplots(1)
figKL,axKL = plt.subplots(1)

L_list = np.array([60,80,100])
N_list = L_list**2

alphamin = 0 # order of quantile for P(r)
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
    optfolder = optfolder0 + f'L{L}/T{T:.2f}/'
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
      print(f'{L=},{T=:.2f} failed')
# sys.exit()
for L_id,L in enumerate(L_list):
  N = L**2
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
else:
  ax.set_ylim(-0.02,1.02)
ax.set_ylabel(r'BID/N')
# ax.set_title(f'Ising triangular')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.grid()
ax.legend(loc='center right')
fig.savefig(figsfolder + f'Triang-Td0.pdf',
            bbox_inches='tight',
            )  

box = axa.get_position()
axa.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axa.legend(loc='center left', bbox_to_anchor=(1, 0.5))

figa.savefig(figsfolder + f'Triang-Td1.pdf',
            bbox_inches='tight',
            ) 
  
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))

figKL.savefig(figsfolder + f'Triang-TKL.pdf',
            bbox_inches='tight',
            )      