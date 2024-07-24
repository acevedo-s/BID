import matplotlib.pyplot as plt
from dadapy import Hamming
import os,sys
import numpy as np
from scipy.signal import argrelextrema

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
np.set_printoptions(precision=10)
markers = ['x','x','s','s','o','o']

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)


geometry = 'Potts-square'
eps = 1E-7
metric = 'hamming'
Tc = .7448
crossed_distances = 0

L_list = [90,100]
precision_T = 3
# T_list = np.array([.7,.746,1])
T_list = np.array([.740,.742,.746])
fig,ax = plt.subplots(1)
plot_id = 0
Ns = 5000
alpha = .1
plot_quantile = 1
for L_id,L in enumerate(L_list):
  N = 3 * L**2
  for T_id,T in enumerate(T_list):
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(T=T,
                  precision_T=precision_T,
                  L=L,
                  Ns=Ns,
                  )
    # peakid = np.argmax(H.D_probs)
    # peakids = argrelextrema(H.D_probs, np.greater)
    # print(f'{peakids}')
    ax.scatter(H.D_values/N,
                H.D_probs,
                color=colors[plot_id%len(colors)],
                label=f'{T=:.3f}',
                )
    H.set_r_quantile(alpha=alpha)
    if plot_quantile:
      ax.vlines(H.r/N,
              ax.get_ylim()[0],
              ax.get_ylim()[1],
              color=colors[plot_id%len(colors)]
              )
    # for peakid in peakids:
    #   ax.vlines(H.D_values[peakid]/N,
    #             ax.get_ylim()[0],
    #             ax.get_ylim()[1],
    #             color='black',
    #             )
    plot_id += 1



ax.set_ylabel(r'$P(r)$')
ax.set_xlabel('r/N')
ax.set_yscale('log')
box = ax.get_position()
ax.set_position([box.x0+.2, box.y0, box.width * 0.8, box.height])
ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1))
# box = ax2.get_position()
# ax2.set_position([box.x0+.2, box.y0, box.width * 0.8, box.height])
# ax2.legend(loc='lower left', bbox_to_anchor=(1.2, .1))
fig.savefig(figsfolder + f'h_L{L}.png',bbox_inches='tight')




