import matplotlib.pyplot as plt
import sys
from dadapy import Hamming
from dadapy._utils import utils_Ising as ui
import os 
import numpy as np


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
# markers = ['p','p','o','o','x','x']
markers = ['s']

optfolder = 'results/opt/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
crossed_distances = 0

figr,axr = plt.subplots(1)
figKL,axKL = plt.subplots(1)
figs,axs = plt.subplots(1)
figa,axa = plt.subplots(1)

N_list = [4096]
alphamin = float(sys.argv[1])
scale_id = int(sys.argv[2]) # to select alphamax

T_list = np.arange(2,2.2+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.21,2.39+eps,.01))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.4,3,.1)
                         )
                         )

### T_dependence
sigma_T = []
alpha_T = []
rmax_T = []
KL_T = []
rmax_T

alphamax_list = np.arange(.1,.3+eps,.1) # orders of quantile for P(r)

rmaxs = np.empty(shape=(len(alphamax_list)))
sigmas = np.empty(shape=rmaxs.shape)
alphas = np.empty(shape=rmaxs.shape)
logKLs = np.empty(shape=rmaxs.shape)

for N_id,N in enumerate(N_list):
  L = int(np.round(np.sqrt(N)))
  k_list = [L]
  for k_id, k in enumerate(k_list):
    for T_id,T in enumerate(T_list):
      for alphamax_id,alphamax in enumerate(alphamax_list):
        outputfile = optfolder + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
        A = np.loadtxt(outputfile,delimiter=',')
        rmaxs[alphamax_id]  = A[0]
        sigmas[alphamax_id]  = A[1]
        alphas[alphamax_id]  = A[2]
        logKLs[alphamax_id]  = A[3]
      # minKL_id = np.where(np.isclose(logKLs,np.nanmin(logKLs)))[0][0]
      # if rmaxs[scale_id] == 1:
      #   scale_id += 1
      sigma_T.append(sigmas[scale_id])
      alpha_T.append(alphas[scale_id])
      rmax_T.append(rmaxs[scale_id])
      KL_T.append(logKLs[scale_id])
    sigma_T = np.array(sigma_T)
    rmax_T = np.array(rmax_T)
    axs.plot(T_list,sigma_T/N,'o-')
    axKL.plot(T_list,KL_T)
    axa.plot(T_list,alpha_T)
    axr.plot(T_list,rmax_T/T)
    


# axs.plot(T_list,
#          s_list,
#          color='black',
#          label='s_exact'
#          )
# print(sigma/k)
# axh.set_xlabel('r')
# axh.set_yscale('log')
# figh.savefig(figsfolder + 'Th.pdf')

axKL.set_ylabel('ln(KL)')
# axKL.set_xlabel(r'$\alpha_r$')
axKL.set_xlabel(r'$T$')
# box = axKL.get_position()
# axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figKL.savefig(figsfolder + 'TKL.png',bbox_inches='tight')

axs.set_ylabel(r'$\sigma/k^2$')
# axs.set_xlabel(r'$\alpha_r$')
axs.set_xlabel(r'$T$')
# axs.legend(loc='lower right')
figs.savefig(figsfolder + 'Tsigma.png')

axa.set_ylabel(r'$\alpha$')
# axa.set_xlabel(r'$\alpha_r$')
axa.set_xlabel('T')
figa.savefig(figsfolder + 'Talpha.png')

figr.savefig(figsfolder + 'Trmax.png')

      