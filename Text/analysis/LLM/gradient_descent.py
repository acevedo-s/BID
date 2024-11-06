from parameters import * 
os.environ["JAX_ENABLE_X64"] = "True"
from dadapy import Hamming
from dadapy._utils.stochastic_minimization_hamming import BID
import os,sys
from time import time 
import numpy as np
from datetime import datetime
from jax import numpy as jnp
_datetime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
print(_datetime)

sys.path.append(f'../../../')
from BID_Optimizer.BID_optimizer import *

eps = 1E-7
os.makedirs(optfolder0,exist_ok=True)
### PARAMETERS
l_id = int(sys.argv[6])
print(f'{l_id=}')
alphamax_id = int(sys.argv[7])
print(f'{alphamax_id=}')
alphamin_id = int(sys.argv[8])
print(f'{alphamin_id=}')

# alphamax_list = np.flip(np.linspace(.15,.7,15))
# alphamax_list = np.arange(.15,.95+eps,.05)
alphamax_list = np.array([.7])
alphamin_list = np.array([1E-5])#,1E-1])
print(f'{alphamax_list=}')
print(f'{alphamin_list=}')

alphamax = alphamax_list[alphamax_id]
print(f'{alphamax=:.5f}')
alphamin = alphamin_list[alphamin_id]
print(f'{alphamin=:.5f}')

if l_id == 24 and LLM == 'OPT':
  emb_dim = 512
else:
  emb_dim = 1024

try:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
except:
   task_id = 10

sub_length = 10*task_id

H = Hamming()
H.D_histogram(k=sub_length,
              t=l_id,
              resultsfolder=histfolder)
B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        )

B.set_idmin()
B.set_idmax()
B.truncate_hist()
B.set_filepaths()

Nsteps = int(1E1)
# delta = jnp.double([500.,2/1000.])
delta = jnp.double([1/10.,1/10.]) 

params = {
  'L':B.L,
  'remp':B.remp,
  'Pemp':B.Pemp,
  'Nsteps':Nsteps
}
vars = {
  'delta' : delta
}
opt = Optimizer(params,vars)
opt = opt.create(params,vars)
opt.params
opt = set_initial_condition(opt)
print(f'{opt.vars["logKLs0"]=}')
print(f'{opt.vars["ds"]=}')
opt = minimize_loss(opt)
print(f'{opt.vars["ds"]=}')
print(f'{opt.vars["logKL"]=}')


