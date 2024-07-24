import numpy as np
import sys,os
import matplotlib.pyplot as plt
from dadapy import Hamming

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

eps = 1E-7
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
crossed_distances = 0

figh,axh = plt.subplots(1,)
T_list = np.flip(np.array([1.8,2,2.27,2.28,2.3,4]))
L_list = [100]
Ns = 5000
for L_id,L in enumerate(L_list):
  for T_id,T in enumerate(T_list):
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
    T=T,
    L=L,
    Ns=Ns,
    )
    axh.plot(H.D_values/L**2,
            H.D_probs,
            'x',
            label=f'{T=:.2f}',
            )

axh.set_yscale('log')
box = axh.get_position()
axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axh.grid(True,zorder=0)
axh.set_xlabel('r/N')
axh.set_ylabel('P(r)')
figh.savefig(figsfolder + 'hists.pdf',
             bbox_inches='tight',
             )
