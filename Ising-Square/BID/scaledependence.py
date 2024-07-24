import matplotlib.pyplot as plt
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
np.set_printoptions(precision=3)
markers = ['p','o','x','^','s']


histfolder = f'../distances/results/hist/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
metric = 'hamming'
crossed_distances = 0


L = int(sys.argv[1])
Nsteps = int(1E6)
delta = 5E-4
alphamin = 0
plot_id = 0
T_list = np.flip(
  np.array([
    1.7,
    1.8,
    2.3,
    3,
    4,
    ]
  )
)
alphamax_list = np.arange(.1,.45+eps,.05) # orders of quantile for P(r)
cut = len(alphamax_list)


fig,ax = plt.subplots(1)
figa,axa = plt.subplots(1)
figkl,axkl = plt.subplots(1)
figr,axr = plt.subplots(1)

log_scale = 0
plot_id = 0
Ns = 5000
alpha_flag = 0
start = 0
seed_list = list(range(1,1+1))
sigma_list = np.empty(shape=(len(T_list),
                            len(seed_list),
                            len(alphamax_list)
                            ),
                      )
rmax_list = np.empty(shape=sigma_list.shape)
logKL_list = np.empty(shape=sigma_list.shape)
alpha_list = np.empty(shape=sigma_list.shape)

for T_id,T in enumerate(T_list):
  for seed_id,seed in enumerate(seed_list):
      for alphamax_id,alphamax in enumerate(alphamax_list):
        H = Hamming(crossed_distances=crossed_distances)
        H.D_histogram(
                      T=T,
                      L=L,
                      Ns=Ns,
                      resultsfolder=histfolder,
                      )
        optfolder0 = f'results/opt/L{L}/T{T:.2f}/'
        B = BID(H,
                alphamin=alphamin,
                alphamax=alphamax,
                seed=seed,
                delta=delta,
                Nsteps=Nsteps,
                optfolder0=optfolder0,
                )
        (rmax_list[T_id,seed_id,alphamax_id],
        sigma_list[T_id,seed_id,alphamax_id],
        alpha_list[T_id,seed_id,alphamax_id],
        logKL_list[T_id,seed_id,alphamax_id],) = B.load_results()

for T_id, T in enumerate(T_list): 
  for seed_id, seed in enumerate(seed_list):
    lbl = f'{T=:.1f}'
    ax.scatter(alphamax_list,
            sigma_list[T_id,seed_id,:]/L**2,
            marker=markers[plot_id%len(colors)],
            color=colors[plot_id%len(colors)],
            label=lbl,
            )
    ax.plot(alphamax_list,
            alpha_list[T_id,seed_id,:]/L**2,
          #  sstd[T_id,:,scale_id]/N_list,
            '-',
            color=colors[plot_id%len(colors)],
            )
    axkl.scatter(alphamax_list,
            logKL_list[T_id,seed_id,:],
            marker=markers[plot_id],
            color=colors[plot_id],
            label=lbl,
            )
    plot_id += 1
ax.set_ylabel(f"BID/N")
ax.set_xlabel(r'order of quantile ($\alpha^*$)')
if log_scale:
  ax.set_yscale('log')
  ax.set_xscale('log')
ax.grid(True)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(ncol=2)
fig.savefig(figsfolder+f'ID_scale_dependence_{L}.pdf',bbox_inches='tight')

axkl.legend()
axkl.set_xlabel(r'order of quantile ($\alpha^*$)')
axkl.set_ylabel(r'$log{(KL)}$')
figkl.savefig(figsfolder+f'KL_scale_dependence_{L}.pdf',bbox_inches='tight')