import sys,os
sys.path.append('../LLM/')
sys.path.append('../../../')
from paths import *
import pyFCI
import numpy as np
from time import time 

start = time()

from parameters import *
from utils import *
import torch
from time import time
torch.set_printoptions(precision=32,sci_mode=False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'{device=}')
print(f'{sys.argv}')
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
assert layer_id in layer_ids
sub_length = int(sys.argv[7])
print(f'{sub_length=}')

N_batches = int(os.getenv('N_batches'))
resultsfolder = makefolder(base=f'results/FCI/',
                          create_folder=True,
                          #  precision=2,
                          Nbits=Nbits,
                          batch_randomize=batch_randomize,
                          Ns=N_batches*batch_size,
                          layer_id=layer_id,
                          sub_length=sub_length,
                          )

act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
X = load_activations(N_batches,
                     act_outputfolder,
                     layer_id,
                     sub_length=sub_length,
                     ).numpy()
Ns0 = X.shape[0]
X = formatting_activations(X,sub_length,Ns0,layer_normalize=0)

if Nbits == 1:
  X = np.round(np.sign(X))
  X[np.where(np.isclose(X,0))] = -1 # just in case

print(f'{X[0]=}')
X = pyFCI.center_and_normalize(X)
fci = pyFCI.FCI(X)
d,x0,err = pyFCI.fit_FCI(fci)
# np.savetxt(resultsfolder + 'fci.txt',X=fci)
np.savetxt(resultsfolder + 'FCI_fit.txt',X=[d,x0,err])

print(f'{d=}')
print(f'{x0=}')
print(f'{err=}')
print(f'this took {(time()-start)/60:.1f} mins')