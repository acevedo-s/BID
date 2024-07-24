import matplotlib.pyplot as plt
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *

def s_exact_chain_PBC(beta, L):
  """
  Exact entropy per site of the 1D ferromagnetic Ising chain with
  periodic boundary conditions and all couplings equal to 1
  """
  partial_beta_log_Z = np.tanh(beta) + 1/(np.sinh(beta) * np.cosh(beta)) * \
                    (1/(1+np.cosh(beta)/np.sinh(beta)))**L 
  log_Z = np.log(2) + np.log(np.cosh(beta)) + (1/L) * np.log(1 + (np.tanh(beta))**L)
  return  - beta * partial_beta_log_Z + log_Z



rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)
markers = ['p','o','h','^','s']
plot_id = 0

figsfolder = f'results/figs/'
resultsfolder = 'results/'

os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
metric = 'hamming'
crossed_distances = 0


L = 10000
R_list = np.arange(1,10+1,dtype=int)
R_list = np.concatenate((R_list,10*np.arange(2,10+1,dtype=int)))
N_list = L*R_list
T_list = np.flip(np.array([2]))

Nsteps = int(1E6)
delta = 5E-4
seed = 111
alphamin = 0# 1E-3

fig,ax = plt.subplots(1)
figa,axa = plt.subplots(1)
figkl,axkl = plt.subplots(1)
figr,axr = plt.subplots(1)

log_scale = 1
Ns = 2500
start = 0
alphamax_list = np.array([.2,.3,.4,.5,.6,.7,.8,.9,1.])[::2]
sigma_list = np.empty(shape=(len(T_list),
                             len(R_list),
                            len(alphamax_list)
                            ),
                      )
logKL_list = np.empty(shape=sigma_list.shape)
alpha_list = np.empty(shape=sigma_list.shape)
rmax_list = np.empty(shape=sigma_list.shape)
for T_id,T in enumerate(T_list):
  for R_id,R in enumerate(R_list):
    for alphamax_id,alphamax in enumerate(alphamax_list):
      H = Hamming()
      optfolder0 = f'results/L{L}/T{T:.2f}/Ns{Ns}/'
      optfolder0 += f'R{R}/'
      B = BID(H,
              alphamin=alphamin,
              alphamax=alphamax,
              seed=seed,
              delta=delta,
              Nsteps=Nsteps,
              optfolder0=optfolder0,
              )
      (rmax_list[T_id,R_id,alphamax_id],
      sigma_list[T_id,R_id,alphamax_id],
      alpha_list[T_id,R_id,alphamax_id],
      logKL_list[T_id,R_id,alphamax_id],) = B.load_results()

for alphamax_id,alphamax in enumerate(alphamax_list):
  for T_id, T in enumerate(T_list):                                            
    IDlabel = f'{alphamax=:.2f}'
    ax.plot(N_list,
            sigma_list[T_id,:,alphamax_id]/N_list,
            'o-',
            label=IDlabel,
            )
    axa.plot(N_list,
          alpha_list[T_id,:,alphamax_id],
          '-o',
          color=colors[plot_id%len(colors)],
          label=IDlabel,
           )
    axkl.plot(N_list,
            logKL_list[T_id,:,alphamax_id],
            'o-',
            label=IDlabel,
            )
    np.savetxt(resultsfolder + f'BID_T{T}_alphamax_{alphamax_list[alphamax_id]:.5f}.txt',
           np.vstack((N_list,
                      sigma_list[T_id,:,alphamax_id],
                      sigma_list[T_id,:,alphamax_id]/N_list,
                      logKL_list[T_id,:,alphamax_id],
                      alpha_list[T_id,:,alphamax_id],
                      )
                    ).T,
            fmt="%.4f"
          )
    plot_id += 1

ax.set_ylabel(f"BID/N")
ax.set_xlabel(r'$N$')
if log_scale:
  ax.set_yscale('log')
  ax.set_xscale('log')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title('Ising Chain')
fig.savefig(figsfolder+f'Chain-d0Nscaling.pdf',bbox_inches='tight')
# ###axa
# axa.legend(ncol=2)
axa.set_title('Ising square')
axa.set_ylabel(r"$d_1$")
axa.set_xlabel(r'$N$')

box = axa.get_position()
axa.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axa.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axa.set_title('Ising Chain')
figa.savefig(figsfolder+f'Chain-d1Nscaling.pdf',bbox_inches='tight')



axkl.set_ylabel(r"$\log{(KL)}$")
axkl.set_xlabel(r'$N$')
box = axkl.get_position()
axkl.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axkl.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axkl.set_title('Ising Chain')
figkl.savefig(figsfolder+f'Chain-KLNscaling.pdf',bbox_inches='tight')

# figr.savefig(figsfolder+f'Square-rstarNscaling.pdf',bbox_inches='tight')