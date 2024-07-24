import matplotlib.pyplot as plt
import sys
from dadapy import Hamming
from dadapy._utils import utils_Ising as ui
import os 
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition,mark_inset)
# import importlib
# print(importlib.import_module('mpl_toolkits').__path__)

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
  axh = plt.axes([0,0,1,1])
  # Manually set the position and relative size of the inset axes within ax1
  ip = InsetPosition(ax, [0.35,0.27,0.6,0.5])
  axh.set_axes_locator(ip)
  (N_list,
   sigma_list,
   alpha_list,
   KL_list) = np.loadtxt(f'{outputfolder}/T{T:.3f}.txt',
                                 unpack=True)
  IDlabel = f'{T=:.1f}'
  ax.scatter(np.log2(N_list),
          sigma_list/N_list,
          marker='s',
          edgecolor='black',
          color=colors[plot_id],
          label=f'{T=:.1f}',
          zorder=2,
          )
  if T==1.9:
    print(sigma_list)
  ax.plot(np.log2(N_list),
          sigma_list/N_list,
          '-',
          color=colors[plot_id],
          zorder=1,
          )
  ### model validation
  for N_id,N in enumerate(N_list):
    (remp,
    Pemp,
    Pmodel) = np.loadtxt(f'{outputfolder}/model_validation_N{int(N):d}_T{T:.2f}.txt',
                                  unpack=True)
    axh.plot(remp/N,
            Pemp,
            'o',
            # label=f'{int(N):d}',
            color=colors[(N_id+1)%len(colors)],
            )
    axh.plot(remp/N,
            Pmodel,
            color='black',
            # label=r'$P_{model}$'
            )
  ax.set_ylabel(r"$BID/N$")
  ax.set_xlabel(r'$\log_2{N}$')  
  # ax.legend(loc='lower left')
  # axh.legend(prop={'size': 15})
  # box = axh.get_position()
  # axh.set_position([box.x0, box.y0, box.width, box.height])
  # axh.legend(loc='upper center', 
  #            bbox_to_anchor=(.5, 1.3),
  #            ncol=len(N_list),
  #            )
  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width, box.height])
  ax.legend(loc='upper center', 
            bbox_to_anchor=(.5, 1.3),
            )
  # plot_id += 1
  if T > 2.5:
    ymax = np.max(sigma_list/N_list) + .05
    ax.set_ylim(-.02,ymax )
    ax.set_yticks(np.arange(0,ymax+eps,.1))
  elif T< 2.2:
    # ymax = np.max(sigma_list/N_list) + .01
    ymax = .11
    ymin = -.35
    ax.set_ylim(ymin,ymax )
    ax.set_yticks(np.arange(0,ymax+eps,.05))
  else:
    # ymax = np.max(sigma_list/N_list) + .03
    ymax = .17
    ymin = -.35
    ax.set_ylim(ymin,ymax )
    ax.set_yticks(np.arange(0,ymax+eps,.05))
  # ax.yaxis.set_ticks_position('both')
  axh.set_yscale('log')
  axh.set_xlabel(r'$r/N$')
  axh.set_ylabel(r'$P(r)$')


figname = f'all_N.pdf'
plt.tight_layout()
fig.savefig(figsfolder+figname)