import matplotlib.pyplot as plt
import sys
from dadapy import Hamming
from dadapy._utils import utils_Ising as ui
import os 
import numpy as np

#for fancy plotting
rcpsize = 22
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=3)
# markers = ['p','p','o','o','x','x']
markers = ['s']
outputfolder = f'results/data/'
os.makedirs(outputfolder,exist_ok=True)
optfolder = f'results/opt/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
metric = 'hamming'
crossed_distances = 0

alphamin = 1E-3
plot_id = 0
T_list = np.array([3,2.3,2])
# T_list = np.flip(T_list)

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15,10))
fig, ((ax1, ax2,ax3)) = plt.subplots(1,3,figsize=(14,5))
ax_list = [ax1,ax2,ax3]
for T_id,T in enumerate(T_list):
  ax = ax_list[T_id]
  ax.grid(True,zorder=0)
  ax.set_axisbelow(True)
  (N_list,
  sigma_list,
  alpha_list,
  KL_list) = np.loadtxt(f'{outputfolder}/T{T:.3f}.txt',
                                 unpack=True)
  IDlabel = f'{T=:.1f}'
  ax.scatter(N_list,
          sigma_list/N_list,
          marker='s',
          edgecolor='black',
          color=colors[plot_id],
          label=f'{T=:.1f}',
          zorder=2,
          )
  ax.plot(N_list,
          sigma_list/N_list,
          '-',
          color=colors[plot_id],
          zorder=1,
          )
  ax.set_ylabel(r"$BID/N$")
  ax.set_xlabel(r'$N$')  
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width, box.height])
  ax.legend(loc='upper center', 
            bbox_to_anchor=(.5, 1.3),
            )
  ax.set_yscale('log')
  ax.set_xscale('log')
figname = f'logplot.pdf'
plt.tight_layout()
fig.savefig(figsfolder+figname)