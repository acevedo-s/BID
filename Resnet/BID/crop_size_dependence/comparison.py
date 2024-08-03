import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *
from R.relative_depth import * 

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
markers = ['p','o','h','^','s']
plot_id = 0

figsfolder = f'results/figs/'
BIDfolder = f'results/'
os.makedirs(BIDfolder,exist_ok=True)
os.makedirs(figsfolder,exist_ok=True)

delta = 5E-4
Nsteps = int(1E6)
seed = 1

model_id = 0
model_name,W_model = model_list[model_id]

layer_names = layers_dict[model_name]


indices = [0,2,3,5,7]
layer_names = [layer_names[idx] for idx in indices]
relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in indices]

alphamin = float(sys.argv[1]) # 0.01
alphamax = float(sys.argv[2]) # 0.1, ,...,0.5

for layer_id,layer_name in enumerate(layer_names):
  plot_id = 0
  figs,axs = plt.subplots(1)
  axs.set_title(r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}')
  ### shuffled
  scrop_sizes,smu,sstd = np.loadtxt(fname=f'{BIDfolder}patches/{layer_name}_alphamax{alphamax:.5f}.txt',unpack=True)  
  slbl = 'patches'
  axs.scatter(scrop_sizes,
            smu[:],
            marker=markers[plot_id],
            edgecolor='black',
            color=colors[plot_id%len(colors)],
            zorder=1,
            )
  axs.plot(scrop_sizes,
            smu[:],
            '--',
            label=slbl,
            color=colors[plot_id%len(colors)],
            zorder=0
            )
  plot_id += 1 

  ### images
  crop_sizes,mu,std = np.loadtxt(fname=f'{BIDfolder}ImageNet/{layer_name}_alphamax{alphamax:.5f}.txt',unpack=True)  
  lbl = 'ImageNet'
  axs.scatter(crop_sizes,
            mu[:],
            marker=markers[plot_id],
            edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            zorder=2
            )
  axs.plot(crop_sizes,
            mu[:],
            '-',
            color=colors[plot_id%len(colors)],
            zorder=1
            )
  # axs.fill_between(crop_sizes[:],
  #             (mu-std)[:],
  #             (mu+std)[:],
  #             color=colors[plot_id%len(colors)],
  #             alpha=.5,
  #             zorder=1,
  #             )

  axs.set_ylabel(r'BID')
  axs.set_xlabel(r'$N_{crop} / N_{tot}$')
  axs.legend(loc='best')
  plt.tight_layout()
  figs.savefig(figsfolder + f'Resnet-comparison-{layer_name}_alphamax{alphamax:.5f}.pdf',
              bbox_inches='tight'
              )