from parameters import * 
import os,sys
from dadapy.hamming import *
from time import time 
import numpy as np
from datetime import datetime
sys.path.append('../../')
from paths import *

_datetime = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
print(_datetime)

start = time()

eps = 1E-7
os.makedirs(optfolder0,exist_ok=True)
### PARAMETERS
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
alphamax_id = int(sys.argv[7])
print(f'{alphamax_id=}')
alphamin_id = int(sys.argv[8])
print(f'{alphamin_id=}')

# alphamax_list = np.flip(np.linspace(.15,.7,15))
# alphamax_list = np.arange(.15,.95+eps,.05)
alphamax_list = np.array([.4,.5,.6,.7])
alphamin_list = np.array([1E-5])#,1E-1])
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

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
sub_length = 10*task_id

N_batches = 50
histfolder = makefolder(base='results/BID/hist/',
                        LLM=LLM,
                        corpus=corpus,
                        layer_id=layer_id,
                        sub_length=sub_length,
                        Nbits=Nbits,
                        batch_randomize=batch_randomize,
                        Ns=N_batches*batch_size,
                        )
H = Hamming()
H.D_histogram(resultsfolder=histfolder)
optfolder0 += f'sublength{sub_length}/layer_id{layer_id}/'
B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        export_logKLs=export_logKLs,
        L=Nbits*emb_dim*sub_length,
        )
B.computeBID()

print(f'this took {(time()-start)/60.} m')