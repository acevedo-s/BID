import matplotlib.pyplot as plt
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *


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

histfolder = f'../distances/results/hist/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
metric = 'hamming'
crossed_distances = 0


L_list = np.arange(30,100+1,10,dtype=int)
Nsteps = int(1E6)
delta = 5E-4
N_list = L_list**2
alphamin = 0# 1E-3
plot_id = 0
T_list = np.flip(
  np.array([1.7,1.8,2.3,3,4]
                  )
)
fig,ax = plt.subplots(1)
figa,axa = plt.subplots(1)
figkl,axkl = plt.subplots(1)
figr,axr = plt.subplots(1)

log_scale = 1
plot_id = 0
Ns = 5000
alpha_flag = 0
start = 0
seed_list = list(range(1,1+1))
alphamax_list = np.array([.1])
sigma_list = np.empty(shape=(len(T_list),
                            len(L_list),
                            len(seed_list),
                            len(alphamax_list)
                            ),
                      )
logKL_list = np.empty(shape=sigma_list.shape)
alpha_list = np.empty(shape=sigma_list.shape)
rmax_list = np.empty(shape=sigma_list.shape)

for T_id,T in enumerate(T_list):
  for L_id,L in enumerate(L_list):
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
        (rmax_list[T_id,L_id,seed_id,alphamax_id],
        sigma_list[T_id,L_id,seed_id,alphamax_id],
        alpha_list[T_id,L_id,seed_id,alphamax_id],
        logKL_list[T_id,L_id,seed_id,alphamax_id],) = B.load_results()

scale_id = 0
seed_id = 0
for T_id, T in enumerate(T_list):                                            
  IDlabel = f'{T=:.1f}'
  ax.scatter(N_list,
          sigma_list[T_id,:,seed_id,scale_id]/N_list,
          marker=markers[plot_id%len(colors)],
          edgecolors='black',
          color=colors[plot_id%len(colors)],
          label=IDlabel,
          s=80,
           )
  ax.plot(N_list,
          sigma_list[T_id,:,seed_id,scale_id]/N_list,
        #  sstd[T_id,:,scale_id]/N_list,
          '-',
          color=colors[plot_id%len(colors)],
           )
  axa.scatter(N_list,
          alpha_list[T_id,:,seed_id,scale_id],
          marker=markers[plot_id%len(colors)],
          edgecolors='black',
          color=colors[plot_id%len(colors)],
          label=IDlabel,
          # s=80
           )
  axa.plot(N_list,
          alpha_list[T_id,:,seed_id,scale_id],
        #  sstd[T_id,:,scale_id]/N_list,
          '-',
          color=colors[plot_id%len(colors)],
           )
  axkl.scatter(N_list,
          logKL_list[T_id,:,seed_id,scale_id],
          marker=markers[plot_id%len(colors)],
          edgecolors='black',
          color=colors[plot_id%len(colors)],
          label=IDlabel,
          # s=80
           )
  axr.scatter(N_list,
        rmax_list[T_id,:,seed_id,scale_id],
        marker=markers[plot_id%len(colors)],
        edgecolors='black',
        color=colors[plot_id%len(colors)],
        label=IDlabel,
        # s=80
          )
  plot_id += 1

ax.set_ylabel(f"BID/N")
ax.set_xlabel(r'$N$')
if log_scale:
  ax.set_yscale('log')
  ax.set_xscale('log')
# ax.grid(True)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(ncol=2)
ax.set_title('Ising square')
fig.savefig(figsfolder+f'Square-d0Nscaling.pdf',bbox_inches='tight')
###axa
axa.legend(ncol=2)
axa.set_title('Ising square')
axa.set_ylabel(r"$d_1$")
axa.set_xlabel(r'$N$')
# axa.hlines(2,
#            axa.get_xlim()[0],
#            axa.get_xlim()[1],
#            color='black',
#            linestyles='dashed')
figa.savefig(figsfolder+f'Square-d1Nscaling.pdf',bbox_inches='tight')

figkl.savefig(figsfolder+f'Square-KLNscaling.pdf',bbox_inches='tight')

figr.savefig(figsfolder+f'Square-rstarNscaling.pdf',bbox_inches='tight')