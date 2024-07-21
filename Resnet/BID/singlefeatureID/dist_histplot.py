import numpy as np
import sys,os
sys.path.append('../../')
import matplotlib.pyplot as plt
from dadapy import Hamming
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

afolder = f'/scratch/sacevedo/Imagenet2012/train/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'


model_id = 0
model_name,W_model = model_list[model_id]
distfolder0 = f'results/{model_name}/'
layer_names = layers_dict[model_name][1:2] # input left aside
optfolder = f'results/opt/{model_name}/'

ED_list = []
Nf_list = []
for layer_id,layer_name in enumerate(layer_names):
    afolder = f'/scratch/sacevedo/Imagenet2012/train/{class_dict["vizsla"]}/a/'
    a_shape = np.loadtxt(afolder + f'{layer_name}/a_shape.txt',dtype='int')
    # print(a_shape)
    if layer_name != 'flatten':
      Nf_list.append(a_shape[1])
      ED_list.append(a_shape[2]*a_shape[3])
    else:
      Nf_list.append(1)
      ED_list.append(a_shape[1])
ED_list = np.array(ED_list)
Nf_list = np.array(Nf_list)

class_list = list(class_dict.keys())[:1]
assert len(ED_list)==len(layer_names)
crossed_distances = 0

figh,axh = plt.subplots(1)
# figm,axm = plt.subplots(1)
plt.xticks(rotation=90)
# normalize_mu = 1

# mus = np.empty(shape=(len(class_list),
#                       len(layer_names))
#                       )
# stds = np.empty(shape=mus.shape)

Nfs = range(8)
for key_id,key in enumerate(class_list):
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  Ns = len(files)
  distfolder = distfolder0 + f'{key}/'
  for layer_id,layer_name in enumerate(layer_names):
    N = ED_list[layer_id]
    Nf = Nf_list[layer_id]
    for f_id in Nfs:
      H = Hamming(crossed_distances=crossed_distances)
      H.D_histogram(
        L=N,
        Ns=Ns,
        t=f_id,
        resultsfolder=distfolder+f'hist/{layer_name}/',
        )
      H.compute_moments()
      # # mus[key_id,layer_id] = H.D_mu_emp
      # # stds[key_id,layer_id] = np.sqrt(H.D_var_emp)
      
      ### optimization
      alphamin = 0.001
      outputfile = optfolder + f'{key}/{layer_name}/f_id{f_id}/alphamin{alphamin:.3f}.txt'
      A = np.loadtxt(outputfile,delimiter=',')
      rmaxs = A[:,0]
      sigmas = A[:,1]
      alphas = A[:,2]
      logKLs = A[:,3]
      alphamaxs = A[:,4]
      minKL_id = np.where(np.isclose(logKLs,min(logKLs)))[0][0]
      if rmaxs[minKL_id] == 1:
        minKL_id += 1
      sigma = sigmas[minKL_id]
      alpha = alphas[minKL_id]
      rmax = rmaxs[minKL_id]

      H.set_r_quantile(alphamin)
      rmin = H.r
      idmin = H.r_idx
      H.r = None
      H.r_idx = None
      idmax = np.where(H.D_values==rmax)[0][0]
      remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
      Pemp = H.D_probs[idmin:idmax+1]
      Pemp /= np.sum(Pemp)
      Pmodel = jnp.zeros(shape=remp.shape, dtype=jnp.float64)
      Op = Optimizer(
                    remp=remp,
                    sigma_r=jnp.double(sigma),
                    alpha_r=jnp.double(alpha),
                    Pmodel=Pmodel,
                    )
      Op = compute_Pmodel(Op)
      axh.plot(remp/N,
              Pemp,
              'x',
              #  label=f'{layer_id+1}',
              )
      axh.plot(remp/N,
               Op.Pmodel,
               color='black')
      
axh.set_yscale('log')
box = axh.get_position()
axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figh.tight_layout()

axh.set_xlabel('r/N')
axh.set_ylabel('P(r)')
figh.savefig(figsfolder + 'h.png',
             bbox_inches='tight',
             )