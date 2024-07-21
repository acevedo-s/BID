import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *
from dadapy._utils.stochastic_minimization_hamming import *


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
np.set_printoptions(precision=7)
markers = ['s']

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

delta = 5E-4
Nsteps = int(5E5)
seed = 100
crop_size = int(sys.argv[1])
plot_fit = int(sys.argv[2])
alphamin = 0.01


model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'../../distances/results/{model_name}/hist/crop_size{crop_size}/'
optfolder0 = f'results/opt/{model_name}/crop_size{crop_size}/'
class_list = list(class_dict.keys())[:1]
layer_names = layers_dict[model_name][:-1]

l1 = 1
l2 = l1+1
layer_names = layer_names[l1:l2]

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

figh,axh = plt.subplots(1)
for key_id,key in enumerate(class_list):
  print(f'{key}({class_dict[key]}):')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  for layer_id,layer_name in enumerate(layer_names):
    EDfolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/{layer_name}/'
    EDfile = EDfolder + f'a_flatten_shape.txt'
    Ns,N = np.genfromtxt(EDfile,
                        dtype='str',
                        unpack=True).astype(int)
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
    for rmax_id,rmax in enumerate(rmaxs):
      H.set_r(rmax)
      H.set_r_idx()
      rmax = H.r
      idmax = H.r_idx
      H.r = None
      H.r_idx = None
      remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
      Pemp = jnp.array(H.D_probs[idmin:idmax+1], dtype=jnp.float64)
      Pemp /= jnp.sum(Pemp)
      axh.plot(remp/N,
               Pemp,
               'x',
              #  color='black',
               label=r'$P_{emp}$',
               zorder=0,
               )
      Pmodel = jnp.zeros(shape=Pemp.shape, dtype=jnp.float64)
      sigma = sigmas[rmax_id]
      alpha = alphas[rmax_id]
      Op = Optimizer(
                    sigma_r=jnp.double(sigma),
                    alpha_r=jnp.double(alpha),
                    remp=remp,
                    Pemp=Pemp,
                    Pmodel=Pmodel,
                    )
      Op = compute_Pmodel(Op)
      if plot_fit:
        axh.plot(remp/N,Op.Pmodel,label=r'$P_{model}$',zorder=1,color='black')

axh.set_xlabel(r'$r/N$')
axh.set_yscale('log')
figh.savefig(figsfolder + f'model_validation{crop_size}.png')      