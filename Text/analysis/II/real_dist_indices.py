from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *
from dadapy import Data
np.set_printoptions(precision=30)

from time import time
start = time()

eps = 1E-7
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'

a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()

a = formatting_activations(a,sub_length,Ns,layer_normalize)
B = a.shape[0]

data = Data(coordinates=a,maxk=B-1)
data.compute_distances(maxk=B-1)

filename = f'r_dist_indices_sub_length{sub_length}'
distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)
os.makedirs(distfolder,exist_ok=True)
np.savetxt(fname=f'{distfolder}{filename}.txt',
           X=data.dist_indices,
           fmt='%d')

print(f'this took {(time()-start)/60:.2f} m')