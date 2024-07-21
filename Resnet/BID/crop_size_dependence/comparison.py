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

# crop_sizes = [122,244]
crop_step = 28
crop_sizes = np.array([crop_step*i for i in range(1,8+1)])
figsfolder = f'results/figs/'
IDfolder = f'results/ID/'
os.makedirs(IDfolder,exist_ok=True)
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

delta = 5E-4
Nsteps = int(5E5)
seed = 100

model_id = 0
model_name,W_model = model_list[model_id]

layer_names = layers_dict[model_name]
l1 = 0
l2 = -1 #l1 + 2

# this discards flatten:
layer_names = layer_names[l1:l2]
relative_depth_dict[model_name] = relative_depth_dict[model_name][l1:l2]

#selected layers:
# indices = range(0,8)
# indices = [0,2,4]
# indices = [int(sys.argv[1])]
indices = list(range(8))
layer_names = [layer_names[idx] for idx in indices]
relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in indices]

nc = 7
class_list = list(class_dict.keys())
# class_list = class_list[nc:nc+1]
class_list = class_list[:nc]

eps = 1E-7
metric = 'hamming'
crossed_distances = 0


normalize = 0
log_scale = 0

start = 0

N0 = 224**2

for layer_id,layer_name in enumerate(layer_names):
  plot_id = 0
  figs,axs = plt.subplots(1,
                        # figsize=(10,5),
                        )
  sigma_list = np.zeros(shape=(len(class_list),
                             len(layer_names),
                             len(crop_sizes),
                             )
                        )

  axs.set_title(r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}')
  ### shuffled
  scrop_sizes,smu,sstd = np.loadtxt(fname=f'{IDfolder}shuffled/{layer_name}',unpack=True)  
  # if len(indices)>1:
  #   slbl = r'$\tilde{l}/\tilde{L}_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}'
  # else:
  slbl = 'patches'
  axs.scatter(scrop_sizes[start:] / N0,
            smu[start:],
            marker=markers[plot_id],
            edgecolor='black',
            # label=slbl,
            color=colors[plot_id%len(colors)],
            zorder=1,
            )
  axs.plot(scrop_sizes[start:] / N0,
            smu[start:],
            '--',
            label=slbl,
            color=colors[plot_id%len(colors)],
            zorder=0
            )
  # axs.fill_between(scrop_sizes[start:],
  #           smu[start:]-sstd[start:],
  #           smu[start:]+sstd[start:],
  #           color=colors[plot_id%len(colors)],
  #           zorder=0,
  #           ) 
  plot_id += 1 

  ### images
  crop_sizes,mu,std = np.loadtxt(fname=f'{IDfolder}{layer_name}',unpack=True)  
  lbl = 'ImageNet'
  axs.scatter(crop_sizes[start:] / N0,
            mu[start:],
            marker=markers[plot_id],
            edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            zorder=2
            )
  axs.plot(crop_sizes[start:] / N0,
            mu[start:],
            '-',
            color=colors[plot_id%len(colors)],
            zorder=1
            )
  # axs.fill_between(crop_sizes[start:],
  #             (mu-std)[start:],
  #             (mu+std)[start:],
  #             color=colors[plot_id%len(colors)],
  #             alpha=.5,
  #             zorder=1,
  #             )


  if normalize:
    axs.set_ylabel(r'$BID/N_{crop}$')
  else:
    axs.set_ylabel(r'BID')
  axs.set_xlabel(r'$N_{crop} / N_{tot}$')

  if log_scale:
    axs.set_yscale('log')
    axs.set_xscale('log')
  axs.legend(loc='best')
  plt.tight_layout()
  figs.savefig(figsfolder + f'Resnet-comparison{layer_id}_log{log_scale}.pdf',
              bbox_inches='tight'
              )