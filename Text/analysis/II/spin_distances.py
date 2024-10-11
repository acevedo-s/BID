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

a = formatting_activations(a,sub_length,Ns,layer_normalize)

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
distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)
os.makedirs(distfolder,exist_ok=True)
np.savetxt(fname=f'{distfolder}{filename}.txt',
           X=H.distances,
           fmt='%d')

print(f'{layer_id=:d} took {(time()-start)/60:.1f} min')

