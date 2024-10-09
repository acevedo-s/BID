from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *
from dadapy import Data

from time import time
start = time()

eps = 1E-7
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'

a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()

# checking for possible repetitions, note that this function reorders data. 
b = np.unique(a,axis=0)
if b.shape != a.shape: 
  print(f'WARNING!!!: there are repetitions in the real-valued activations')

B,T,E = a.shape
# keeping only sub_length tokens
a = a[:,:sub_length,:]
print(f'{a.shape=}')
# vectorizing activations
a  = np.reshape(a,(B,sub_length*E))

if layer_id == 0:
  a = poor_mans_layer_norm(a,N_batches,batch_size)

data = Data(coordinates=a,maxk=B-1)
data.compute_distances(maxk=B-1)

filename = f'r_dist_indices_sub_length{sub_length}'
os.makedirs(distfolder,exist_ok=True)
np.savetxt(fname=f'{distfolder}{filename}.txt',
           X=data.dist_indices,
           fmt='%d')

print(f'this took {(time()-start)/60:.2f} m')