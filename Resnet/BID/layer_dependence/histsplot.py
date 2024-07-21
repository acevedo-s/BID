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
# print(plt.rcParams.keys())
np.set_printoptions(precision=7)
# markers = ['p','p','o','o','x','x']
markers = ['s']

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'../../B/results/Bd/{model_name}/'
resultsfolder = f'results/opt/{model_name}/'
class_list = list(class_dict.keys())[:1]
EDfile = histfolder + f'ED.txt'
A = np.genfromtxt(EDfile,dtype='str')
ED_list = A[:,1].astype(int)
layer_names = layers_dict[model_name][1:]

l1 = 6
l2 = l1+1
layer_names = layer_names[l1:l2]
ED_list = ED_list[l1:l2]
assert(len(ED_list)==len(layer_names))

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

figh,axh = plt.subplots(1)
alphamin = 0.01
for key_id,key in enumerate(class_list):
  print(f'{key}({class_dict[key]}):')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  Ns = len(files)
  for layer_id,layer_name in enumerate(layer_names):
    N = ED_list[layer_id]
    print(f'{layer_name},{N=}')
    outputfile = resultsfolder + f'{key}/{layer_name}/alphamin{alphamin:.3f}.txt'
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
      Ns=Ns,
      resultsfolder=histfolder+f'{class_dict[key]}/hist/{layer_name}/',
      )
    H.set_r_quantile(alphamin)
    rmin = H.r
    idmin = H.r_idx
    H.r = None
    H.r_idx = None

    A = np.loadtxt(outputfile,delimiter=',')
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
      axh.plot(remp/N,Op.Pmodel,label=r'$P_{model}$',zorder=1,color='black')

axh.set_xlabel(r'$r/N$')
axh.set_yscale('log')
figh.savefig(figsfolder + 'h.pdf')      