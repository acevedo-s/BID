from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *
from dadapy.hamming import Hamming
from time import time
start = time()

neighbour = 1
eps = 1E-7
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'

a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()

# eliminating possible repetitions, note that this reorders data. 
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

# binarization
spins = np.sign(a)
spins[np.asarray(spins == 0).nonzero()] = -1

# distances
H = Hamming(coordinates=spins)
start = time()
H.compute_distances()
H.distances += np.transpose(H.distances) # symmetric... just for format

# exporting
filename = f's_dists_sub_length{sub_length}'
os.makedirs(distfolder,exist_ok=True)
np.savetxt(fname=f'{distfolder}{filename}.txt',
           X=H.distances,
           fmt='%d')

print(f'{layer_id=:d} took {(time()-start)/60:.1f} min')

