import matplotlib.pyplot as plt
import sys,os
sys.path.append('../../')
os.environ['JAX_ENABLE_X64'] = 'True'
from dadapy import Hamming
from dadapy._utils.stochastic_minimization_hamming import BID
import numpy as np
from R.models import *
from R.data import *
from R.relative_depth import * 
from R.paths import *


rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)
markers = ['p','o','h','^','s']
plot_id = 0

crop_step = 28
crop_sizes = np.array([crop_step*i for i in range(2,8+1,2)])
max_crop = crop_sizes[-1]
figsfolder = f'results/figs/patches/'
BIDfolder = f'results/patches/'
distance_folder = "../../distances"
optimization_folder = '../optimize'
os.makedirs(BIDfolder,exist_ok=True)
os.makedirs(figsfolder,exist_ok=True)

delta = 5E-4
Nsteps = int(1E6)
seed = 1

model_id = 0
model_name,W_model = model_list[model_id]
layer_names = layers_dict[model_name]


# this discards flatten:
# layer_names = layer_names[l1:l2]
# relative_depth_dict[model_name] = relative_depth_dict[model_name][l1:l2]

#selected layers:
indices = [0,2,3,5,7]
layer_names = [layer_names[idx] for idx in indices]
relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in indices]

class_list = list(class_dict.keys())
r_ids = np.arange(0,len(class_list),1,dtype=int)
key = 'shuffled'

figKL,axKL = plt.subplots(1)
fig0,ax0 = plt.subplots(1)
fig1,ax1 = plt.subplots(1)

alphamin = float(sys.argv[1]) # 0.01
alphamax = float(sys.argv[2]) # 0.1, ,...,0.5

d0_list = np.zeros(shape=(len(r_ids),
                          len(layer_names),
                          len(crop_sizes),
                          )
                  )
d1_list = np.zeros(shape=d0_list.shape)
logKL_list = np.zeros(shape=d0_list.shape)
rmax_list = np.zeros(shape=d0_list.shape)

for r_id in r_ids:
  for layer_id,layer_name in enumerate(layer_names):
    print(f'{layer_name=}')
    for crop_id,crop_size in enumerate(crop_sizes):
      print(f'{crop_size=}')
      optfolder0 = get_optfolder(optimization_folder,model_name,crop_size,key,layer_name)
      histfolder = get_histfolder(distance_folder,model_name,crop_size,key,layer_name)
      EDfile = get_EDfilename(distance_folder,model_name,layer_name,key)
      Ns,N = np.genfromtxt(EDfile,
                    dtype='str',
                    unpack=True).astype(int)
      H = Hamming()
      H.D_histogram(
                    r_id=r_id,
                    resultsfolder=histfolder,
                    )
      B = BID(H,
              alphamin=alphamin,
              alphamax=alphamax,
              seed=seed,
              delta=delta,
              Nsteps=Nsteps,
              optfolder0=optfolder0,
              )
      (rmax_list[r_id,layer_id,crop_id],
      d0_list[r_id,layer_id,crop_id],
      d1_list[r_id,layer_id,crop_id],
      logKL_list[r_id,layer_id,crop_id]) = B.load_results()

mu_d0 = np.nanmean(d0_list,axis=0)
std_d0 = np.nanstd(d0_list,axis=0)
mu_d1 = np.nanmean(d1_list,axis=0)
std_d1 = np.nanstd(d1_list,axis=0)
mu_logKL = np.nanmean(logKL_list,axis=0)
std_logKL = np.nanstd(logKL_list,axis=0)


for layer_id,layer_name in enumerate(layer_names):
  lbl = r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}'
  ax0.plot(crop_sizes**2 / max_crop**2,
            mu_d0[layer_id,:],
            'o-',
            label=lbl,
            color=colors[plot_id%len(colors)],
  )
  ax1.plot(crop_sizes**2 / max_crop**2, 
            mu_d1[layer_id,:],
            '-o',
            # marker=markers[plot_id],
            # edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
  )
  axKL.plot(crop_sizes**2 / max_crop**2,
            mu_logKL[layer_id,:],
            # marker=markers[plot_id],
            # edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            )
  plot_id += 1
  np.savetxt(fname=f'{BIDfolder}/{layer_name}_alphamax{alphamax:.5f}.txt',
             X=np.transpose([crop_sizes**2 / max_crop**2,
                             mu_d0[layer_id,:],
                             std_d0[layer_id,:]
                             ]
                            )
  )


ax0.set_ylabel(r'BID')
box = ax0.get_position()
ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig0.savefig(figsfolder + f'd0_crop_{alphamax:.5f}.pdf',bbox_inches='tight')

ax1.set_ylabel(r'BID')
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig1.savefig(figsfolder + f'd1_crop_{alphamax:.5f}.pdf',bbox_inches='tight')

axKL.set_ylabel(r'BID')
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figKL.savefig(figsfolder + f'KL_crop_{alphamax:.5f}.pdf',bbox_inches='tight')
