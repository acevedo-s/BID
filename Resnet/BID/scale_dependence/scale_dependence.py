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
eps = 1E-7


class_id = int(sys.argv[1])
layer_id = int(sys.argv[2])

crop_step = 28
crop_sizes = np.array([crop_step*i for i in range(1,8+1)])


figsfolder = f'results/figs/ImageNet/scale_dependence/'
distance_folder = "../../distances"
optimization_folder = '../optimize'
os.makedirs(figsfolder,exist_ok=True)

delta = 5E-4
Nsteps = int(1E6)
seed = 1

model_id = 0
model_name,W_model = model_list[model_id]
layer_names = layers_dict[model_name]
layer_name = layer_names[layer_id]


# this discards flatten:
# layer_names = layer_names[l1:l2]
# relative_depth_dict[model_name] = relative_depth_dict[model_name][l1:l2]

# selecting layers: 
# indices = [0,2,3,5,7]
indices = [0,1]
layer_names = [layer_names[idx] for idx in indices]
relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in indices]

# selecting classes:
nc = 1
class_list = list(class_dict.keys())[:nc]


alphamin_list = np.array([0.01])
alphamax_list = np.arange(0.1,.9 + eps, 0.1)
alphamax_list = np.concatenate((alphamax_list,[.99]))

d0_list = np.zeros(shape=(len(class_list),
                          len(layer_names),
                          len(crop_sizes),
                          len(alphamin_list),
                          len(alphamax_list),
                          )
)
d1_list = np.zeros(shape=d0_list.shape)
logKL_list = np.zeros(shape=d0_list.shape)
rmax_list = np.zeros(shape=d0_list.shape)


for class_id,key in enumerate(class_list):
  print(f'{key=}')
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
      for alphamin_id,alphamin in enumerate(alphamin_list):
        for alphamax_id,alphamax in enumerate(alphamax_list):
          H = Hamming()
          H.D_histogram(
                        Ns=Ns,
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
          (rmax_list[class_id,layer_id,crop_id,alphamin_id,alphamax_id],
          d0_list[class_id,layer_id,crop_id,alphamin_id,alphamax_id],
          d1_list[class_id,layer_id,crop_id,alphamin_id,alphamax_id],
          logKL_list[class_id,layer_id,crop_id,alphamin_id,alphamax_id]) = B.load_results()



fig0,ax0 = plt.subplots(1)
figKL,axKL = plt.subplots(1)

class_id = int(sys.argv[1])
layer_id = int(sys.argv[2])
alphamin_id = 0

for crop_id,crop_size in enumerate(crop_sizes):
    # lbl = r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}'
    lbl= f'{crop_size=}'
    ax0.plot(alphamax_list,
              d0_list[class_id,layer_id,crop_id,alphamin_id,:],
              'o-',
              label=lbl,
              color=colors[plot_id%len(colors)],
              )
    axKL.plot(alphamax_list,
              logKL_list[class_id,layer_id,crop_id,alphamin_id,:],
              '-o',
              color=colors[plot_id%len(colors)],
              label=lbl,
              )
    plot_id += 1

ax0.set_xlabel(r'$\alpha_{max}$')
ax0.set_ylabel(r'$d_0$')
box = ax0.get_position()
ax0.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig0.savefig(figsfolder + f'd0_scale_dependence_{key}_layer_id{layer_id}_crops.pdf',bbox_inches='tight')

axKL.set_xlabel(r'$\alpha_{max}$')
axKL.set_ylabel(r'$logKL$')
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figKL.savefig(figsfolder + f'KL_scale_dependence_{key}_layer_id{layer_id}_crops.pdf',bbox_inches='tight')
