from parameters import * 
from dadapy import Hamming
import os,sys
from dadapy._utils.stochastic_minimization_hamming import *
# from dadapy._utils.gradient_descent_hamming import *
from time import time 
import numpy as np
from datetime import datetime
_datetime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
print(_datetime)

eps = 1E-7
os.makedirs(optfolder0,exist_ok=True)
### PARAMETERS
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
alphamax_id = int(sys.argv[7])
print(f'{alphamax_id=}')
alphamin_id = int(sys.argv[8])
print(f'{alphamin_id=}')
tau = int(sys.argv[9])
print(f'{tau=}')

# alphamax_list = np.arange(.15,.95+eps,.05)
# alphamin_list = np.array([1E-5,1E-1])
alphamax_list = np.array([.2])
alphamin_list = np.array([1E-5])
print(f'{alphamax_list=}')
print(f'{alphamin_list=}')

alphamax = alphamax_list[alphamax_id]
print(f'{alphamax=:.5f}')
alphamin = alphamin_list[alphamin_id]
print(f'{alphamin=:.5f}')

if layer_id == 24 and LLM == 'OPT':
  emb_dim = 512
else:
  emb_dim = 1024

Nsteps = int(5E6)
print(f'{Nsteps=}')
seed = 111
delta = 7E-4

H = Hamming()
H.D_histogram(resultsfolder=histfolder+f'2distances/layer{layer_id}/tau{tau}/')
optfolder0 += f'layer{layer_id}/tau{tau}/'
B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        export_logKLs=export_logKLs,
        L=2*emb_dim
        )
B.computeBID()

