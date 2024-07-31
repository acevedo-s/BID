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
crop_size = int(sys.argv[3])
print(f'{crop_size=}')


figsfolder = f'results/figs/ImageNet/fits/'
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

# selected layers:
# layer_indices = [0,2,3,5,7]
# layer_indices = [0]
# layer_names = [layer_names[idx] for idx in layer_indices]
# relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in layer_indices]
# print(f'{layer_name=}')


class_list = list(class_dict.keys())
key = class_list[class_id]


alphamin_list = np.array([0.01])
alphamax_list = np.arange(0.1,.9 + eps, 0.1)
alphamax_list = np.concatenate((alphamax_list,[.99]))

d0_list = np.zeros(shape=(len(alphamin_list),
                          len(alphamax_list),
                  )
)
d1_list = np.zeros(shape=d0_list.shape)
logKL_list = np.zeros(shape=d0_list.shape)
rmax_list = np.zeros(shape=d0_list.shape)


optfolder0 = get_optfolder(optimization_folder,model_name,crop_size,key,layer_name)
histfolder = get_histfolder(distance_folder,model_name,crop_size,key,layer_name)
EDfile = get_EDfilename(distance_folder,model_name,layer_name,key)
Ns,N = np.genfromtxt(EDfile,
              dtype='str',
              unpack=True).astype(int)


figh,axh = plt.subplots(1)
figKL,axKL = plt.subplots(1)

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
    (rmax_list[alphamin_id,alphamax_id],
    d0_list[alphamin_id,alphamax_id],
    d1_list[alphamin_id,alphamax_id],
    logKL_list[alphamin_id,alphamax_id]) = B.load_results()
    remp,Pemp,Pmodel = B.load_fit()

    # mu_d0 = np.nanmean(d0_list,axis=0)
    # std_d0 = np.nanstd(d0_list,axis=0)
    # mu_d1 = np.nanmean(d1_list,axis=0)
    # std_d1 = np.nanstd(d1_list,axis=0)
    # mu_logKL = np.nanmean(logKL_list,axis=0)
    # std_logKL = np.nanstd(logKL_list,axis=0)

    # lbl = r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}'
    lbl = r'$\alpha_{max}=$' + f'{alphamax:.1f}'
    axh.plot(remp/N,
              Pemp,
              'x',
              label=lbl,
              color=colors[plot_id%len(colors)],
              )
    axh.plot(remp/N,
              Pmodel,
              '-',
              color='black',
              )
    plot_id += 1


axh.set_xlabel(r'r')
axh.set_ylabel(r'P(r)')
box = axh.get_position()
axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figh.savefig(figsfolder + f'{key}_layer_id{layer_id}_cropsize{crop_size}.pdf',bbox_inches='tight')
