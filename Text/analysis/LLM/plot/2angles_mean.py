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

taus = np.array(([9 + 10 * i for i in range(0,29+1,1)]))
t_list = [0]
layer_ids = [0]

angles = np.zeros(shape=(len(taus)))


fig,ax = plt.subplots(1,
                      # figsize=(8,4)
                      )

for batch_shuffle_flag in batch_shuffle_flags:
  if batch_shuffle_flag:
    anglesfolder0 += f'Lconcat{Lconcat}/'
  for layer_id in layer_ids:
    for t_id,t in enumerate(t_list):
      for tau_id,tau in enumerate(taus):
        datafile = get_angles_filename(anglesfolder0,
                                      t,
                                      tau,
                                      layer_id)
        angles[tau_id] = np.mean(np.loadtxt(datafile),axis=0)
        if batch_shuffle_flag:
          lbl = f'C {layer_id}'
          marker = 's'
        else:
          lbl = f'W {layer_id}'
          marker = 'o'
    ax.plot(taus, 
            angles, 
            marker=marker,
            label=lbl,
            )

ax.set_ylabel(r"$\langle Cos(E_t,E_{t+\tau}) \rangle $")
ax.set_xlabel(r'$\tau$')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax.set_title(f'{layer_id=}')
fig.savefig(figsfolder + 'angles_means.png',
            # bbox_inches='tight',
            )