import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *
from dadapy._utils.stochastic_minimization_hamming import *


filesfolder = f'../../results/files/'
model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'results/{model_name}/'
resultsfolder = f'results/opt/{model_name}/'
class_list = list(class_dict.keys())[:1]
l1 = 1
# l1 = int(sys.argv[-1])
l2 = None#l1 + 1
assert l1 > 0 # input left aside
layer_names = layers_dict[model_name][l1:l2] 
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

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

alphamin = 0.001
# alphamax_list = [.1,.2,.3,.4,.5,.6,.7,.8,.9,.95]
alphamax_list = np.arange(.1,1+eps,.1)
alpha0 = 1. # initial slope of ID 
delta = 1E-3
Nsteps = int(1E5)
seed_list = [123]
remove_previous_output = 1

for key_id,key in enumerate(class_list):
  print(f'{key}({class_dict[key]}):')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  Ns = len(files)
  for layer_id,layer_name in enumerate(layer_names):
    N = ED_list[layer_id]
    Nf = Nf_list[layer_id]
    print(f'{layer_name},{N=},Nf={Nf}')
    # Nf = 5
    for f_id in range(Nf):
      print(f'{f_id=:d}')
      outputfile = resultsfolder + f'{key}/{layer_name}/f_id{f_id}/alphamin{alphamin:.3f}.txt'
      os.makedirs(resultsfolder+f'{key}/{layer_name}/f_id{f_id}/',exist_ok=True)
      if remove_previous_output:
        os.system(f'rm -f {outputfile}')
      H = Hamming(crossed_distances=crossed_distances)
      H.D_histogram(
        Ns=Ns,
        t=f_id,
        L=N,
        resultsfolder=histfolder+f'{key}/hist/{layer_name}/',
        )
      H.set_r_quantile(alphamin)
      rmin = H.r
      idmin = H.r_idx
      H.r = None
      H.r_idx = None
      for seed_id,seed in enumerate(seed_list):
        KL_list = []
        sigma_list = []
        alpha_list = []
        rmaxs = []
        for alphamax_id,alphamax in enumerate(alphamax_list):
          H.set_r_quantile(alphamax)
          rmax = H.r
          if rmax == 0:
            continue
          idmax = H.r_idx
          H.r = None
          H.r_idx = None
          remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
          Pemp = jnp.array(H.D_probs[idmin:idmax+1], dtype=jnp.float64)
          Pemp /= jnp.sum(Pemp)
          Pmodel = jnp.zeros(shape=Pemp.shape, dtype=jnp.float64)
          key0 = random.PRNGKey(seed)
          Op = Optimizer(key=key0,
                        sigma=jnp.double(rmax),
                        alpha=jnp.double(alpha0),
                        delta=jnp.double(delta),
                        remp=remp,
                        Pemp=Pemp,
                        Pmodel=Pmodel,
                        Nsteps=Nsteps,
                        )
          Op = minimize_KL(Op)
          print(f'{alphamax=:.2f},{jnp.log(Op.KL)=}')
          print(f'{rmax:d},{Op.sigma:.8f},{Op.alpha:8f},{np.log(Op.KL):.8f},{alphamax:.3f},{rmin:d}',
                file=open(outputfile, 'a'))
