import sys,os
sys.path.append('../../')
from dadapy import Hamming
import numpy as np
from R.models import *
from R.data import *
from dadapy._utils.stochastic_minimization_hamming import *
from R.relative_depth import * 
from time import time
start = time()

crop_size = int(sys.argv[1])
print(f'{crop_size=}')
layer_id = int(sys.argv[2])
try:
  r_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
  r_id = int(sys.argv[3])
# 3=peak 
# 7=previous to flatten

model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'../../distances/results/{model_name}/hist/crop_size{crop_size}/'
optfolder0 = f'results/opt/{model_name}/crop_size{crop_size}/'
helpfolder0 = f'results/opt/{model_name}/crop_size112/'
layer_names = layers_dict[model_name]

layer_name = layer_names[layer_id] 
relative_depth = relative_depth_dict[model_name][layer_id]
print(f'{relative_depth=}')

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

alphamin = 0.01
alphamax_list = np.array([.1,.2,.3])
alpha00 = 1. # initial slope of ID 
delta = 5E-4
Nsteps = int(5E5)
seed = 100
export_output_flag = 1
remove_previous_output = 1

EDfolder = f'/scratch/sacevedo/Imagenet2012/act/shuffled/crop_size{crop_size}/{layer_name}/'
EDfile = EDfolder + f'a_flatten_shape.txt'
Ns,N = np.genfromtxt(EDfile,
                        dtype='str',
                        unpack=True).astype(int)
print(f'{layer_name},{N=}')
optfolder = optfolder0 + f'shuffled/{r_id}/{layer_name}/alphamin{alphamin:.5f}/'
optfolder += f'delta{delta}/seed{seed}/'
optfile = optfolder + 'opt.txt'
os.makedirs(optfolder,exist_ok=True)
H = Hamming(crossed_distances=crossed_distances)
H.D_histogram(
  r_id=r_id,
  resultsfolder=histfolder+f'shuffled/{layer_name}/',
  )
H.compute_moments()
H.set_r_quantile(alphamin)
rmin = H.r
idmin = H.r_idx
H.r = None
H.r_idx = None
KL_list = []
sigma_list = []
alpha_list = []
rmaxs = []
for alphamax_id,alphamax in enumerate(alphamax_list):
  H.set_r_quantile(alphamax)
  rmax = H.r
  idmax = H.r_idx
  H.r = None
  H.r_idx = None
  remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
  Pemp = jnp.array(H.D_probs[idmin:idmax+1], dtype=jnp.float64)
  Pemp /= jnp.sum(Pemp)
  Pmodel = jnp.zeros(shape=Pemp.shape, dtype=jnp.float64)
  key0 = random.PRNGKey(seed)
  sigma0 = H.D_mu_emp
  alpha0 = alpha00
  ### First layers have problems when crop_size is small:
  if layer_id <=3 and crop_size <= 100:
    help_id = layer_id
    helpfolder = helpfolder0 + f'shuffled/{r_id}/{layer_names[help_id]}/alphamin{alphamin:.5f}/'
    helpfolder += f'delta{delta}/seed{seed}/'
    helpfile= helpfolder + 'opt.txt'
    A = np.loadtxt(helpfile,delimiter=',')
    if len(A.shape)==1:
      A = np.vstack([A,A])
    helpsigmas = A[:,1]
    helpalphas = A[:,2]
    helplogKLs = A[:,3]
    helpminKL_id = np.where(np.isclose(helplogKLs,np.nanmin(helplogKLs)))[0][0]
    sigma0 = helpsigmas[helpminKL_id]
    alpha0 = helpalphas[helpminKL_id]
  Op = Optimizer(key=key0,
                sigma=jnp.double(sigma0),
                alpha=jnp.double(alpha0),
                delta=jnp.double(delta),
                remp=remp,
                Pemp=Pemp,
                Pmodel=Pmodel,
                Nsteps=Nsteps,
                )
  nan_counts,inf = check_initial_condition(Op)
  if nan_counts != 0: 
    print('Pmodel gives nan for the initial conditions')
    continue
  if inf != 0: 
    print('KL is infinite for the initial conditions')
    continue
  Op = minimize_KL(Op)
  print(f'{alphamax=:.2f},{Op.sigma:.8f},{Op.alpha:8f},{jnp.log(Op.KL)=}')
  if alphamax_id == 0:
    if export_output_flag:
      if remove_previous_output:
        os.system(f'rm -f {optfile}')
  print(f'{rmax:d},{Op.sigma:.8f},{Op.alpha:8f},{np.log(Op.KL):.8f},{alphamax:.5f}',
        file=open(optfile, 'a'))
  print(f'{Op.acc_ratio=}')
  print(f'this took {(time()-start)/60.:.1f} mins')
