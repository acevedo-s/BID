import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *
from R.relative_depth import * 

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
np.set_printoptions(precision=7)
# plt.xticks(rotation=90)

# markers = ['p','p','o','o','x','x']
markers = ['s']


crop_size = int(sys.argv[1])
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'../../distances/results/{model_name}/hist/crop_size{crop_size}/'
optfolder0 = f'../optimize/results/opt/{model_name}/crop_size{crop_size}/'
layer_names = layers_dict[model_name]
l1 = 0
l2 = None# l1 + 1
nc = 5
layer_names = layer_names[l1:l2]
ED_list = np.zeros(shape=(len(layer_names)),
                          dtype=int
                   )
class_list = list(class_dict.keys())[:nc]

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

figKL,axKL = plt.subplots(1)
figs,axs = plt.subplots(1)
figa,axa = plt.subplots(1)

alphamin = 0.01
sigma_list = np.zeros(shape=(len(class_list),
                             len(layer_names)
                             )
                      )
alpha_list = np.zeros(shape=sigma_list.shape)
KL_list = np.zeros(shape=sigma_list.shape)
normalize = 0
delta = 5E-4
Nsteps = int(5E5)
seed = 100

for key_id,key in enumerate(class_list):
  print(f'{key}({class_dict[key]}):')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  Ns = len(files)
  for layer_id,layer_name in enumerate(layer_names):
    EDfolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/{layer_name}/'
    EDfile = EDfolder + f'a_flatten_shape.txt'
    Ns_act,N = np.genfromtxt(EDfile,
                            dtype='str',
                            unpack=True).astype(int)
    ED_list[layer_id] = N
    print(f'{layer_name},{N=}')
    optfolder = optfolder0 + f'{key}/{layer_name}/alphamin{alphamin:.5f}/'
    optfolder += f'delta{delta}/seed{seed}/'
    optfile = optfolder + 'opt.txt'
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
      Ns=Ns,
      resultsfolder=histfolder+f'{key}/{layer_name}/',
      )
    H.set_r_quantile(alphamin)
    rmin = H.r
    idmin = H.r_idx
    H.r = None
    H.r_idx = None
    A = np.loadtxt(optfile,delimiter=',')
    if len(A.shape)==1:
      A = np.vstack([A,A])
    rmaxs = A[:,0]
    sigmas = A[:,1]
    alphas = A[:,2]
    logKLs = A[:,3]
    minKL_id = np.where(np.isclose(logKLs,min(logKLs)))[0][0]
    sigma_list[key_id,layer_id] = sigmas[minKL_id]
    alpha_list[key_id,layer_id] = alphas[minKL_id]
    KL_list[key_id,layer_id] = logKLs[minKL_id]
mu = np.mean(sigma_list,axis=0)
std = np.std(sigma_list,axis=0)
mu_a = np.mean(alpha_list,axis=0)
std_a = np.std(alpha_list,axis=0)
mu_KL = np.mean(KL_list,axis=0)
std_KL = np.std(KL_list,axis=0)
if normalize:
  mu /= ED_list
  std /= ED_list
# axs.errorbar(relative_depth_dict[model_name],
#              mu,
#              std,
#              capsize=12,
#             #  'o-',
#             label=f'ID',
#             )
plot_id = 5 # for color
axs.scatter(relative_depth_dict[model_name][l1:l2],
          mu,
          marker='o',
          edgecolor='black',
          label=f'BID',
          color=colors[plot_id],
          )
axs.fill_between(relative_depth_dict[model_name][l1:l2],
            mu-std,
            mu+std,
            color=colors[plot_id],
            alpha=.5,
            zorder=0,
            )
axED = axs.twinx()
axED.set_ylabel('N')
axED.plot(relative_depth_dict[model_name][l1:l2],
        ED_list,
        color='black',
        linestyle='dashed',
        label='N',
        )

if normalize:
  axs.set_ylabel(r'BID/N')
else:
  axs.set_ylabel(r'BID')
axs.set_xlabel(f'relative depth')


axs.ticklabel_format(style='sci',
                    axis='y', 
                    scilimits=(5,5), 
                    useMathText=True
                    )
axED.ticklabel_format(style='sci',
                    axis='y', 
                    scilimits=(5,5), 
                    useMathText=True
                    )
axED.legend(loc='upper right')
axs.legend(loc='lower left')
# axs.set_yscale('log')

figs.savefig(figsfolder + 'l_sigma.pdf',bbox_inches='tight')


axa.scatter(relative_depth_dict[model_name][l1:l2],
          mu_a,
          marker='o',
          edgecolor='black',
          label=r'$\alpha$',
          color=colors[0],
          )
axa.set_ylabel(r'$\alpha$')
axa.set_xlabel(f'relative depth')



figa.savefig(figsfolder + 'l_alpha.pdf')
