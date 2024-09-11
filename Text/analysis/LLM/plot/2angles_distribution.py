import sys,os
sys.path.append('LLM/')

import matplotlib.pyplot as plt
import numpy as np
from utils import *

#for fancy plotting
lsize = 20
plt.rcParams['xtick.labelsize']=lsize
plt.rcParams['ytick.labelsize']=lsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = lsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=3)
# markers = ['p','p','o','o','x','x']
markers = ['p','^','s','o']
eps = 1E-7


LLM = 'Pythia'
corpus = 'Wikitext'
batch_shuffle_flags = [0,1]
Lconcat = 150

wd = os.environ['WORK']
path0 = "/sacevedo/Data/Text/"
figsfolder = f'results/{corpus}/{LLM}/angles/'
os.makedirs(figsfolder,exist_ok=True)
anglesfolder0 = wd + path0 + f'{corpus}/{LLM}/angles/'

tau_list = np.array(([9 + 10 * i for i in range(0,29+1,10)]))
print(f'{tau_list=}')
t_list = [0]
layer_ids = [24]

Ns = 5000
prods = np.zeros(shape=(Ns,len(tau_list)))


fig,ax = plt.subplots(1,
                      # figsize=(8,4)
                      )

for batch_shuffle_flag in batch_shuffle_flags:
  if batch_shuffle_flag:
    anglesfolder0 += f'Lconcat{Lconcat}/'
  for layer_id in layer_ids:
    for t_id,t in enumerate(t_list):
      for tau_id,tau in enumerate(tau_list):
        datafile = get_angles_filename(anglesfolder0,
                                      t,
                                      tau,
                                      layer_id)
        prods[:,tau_id] = np.loadtxt(datafile)
        hist,bin_edges = np.histogram(prods[:,tau_id],density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])  # Calculate bin centers
        if tau_id != 0:
          lbl = None
        if batch_shuffle_flag:
          lbl = f'Concat {tau=}'
          color = 'blue'
        else:
          lbl = f'Wikitext {tau=}'
          color = 'black'
        plt.plot(bin_centers, 
                hist, 
                '-o', 
                 label=lbl,
                 color=color,
                )

ax.set_ylabel(r"$Prob(E_t \cdot E_{t+\tau})$")
ax.set_xlabel(r'$E_t \cdot E_{t+\tau}$')
ax.legend()

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


ax.set_title(f'{layer_id=}')
# plt.tight_layout()
fig.savefig(figsfolder + 'angles_distribution.png',
            # bbox_inches='tight',
            )