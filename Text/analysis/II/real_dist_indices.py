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

N_batches = int(sys.argv[7])
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()

a = formatting_activations(a,sub_length,Ns,layer_normalize)
B = a.shape[0]

data = Data(coordinates=a,maxk=B-1)
data.compute_distances(maxk=B-1)


distfolder = get_distfolder(corpus,
                            LLM,
                            layer_id,
                            layer_normalize,
                            Ntokens=Ntokens,
)
os.makedirs(distfolder,exist_ok=True)
# filename = f'r_dist_indices_sub_length{sub_length}'
# np.savetxt(fname=f'{distfolder}{filename}.txt',
#            X=data.dist_indices,
#            fmt='%d')
hist_filename = f'histogram_sub_length{sub_length}'
bin_edges_filename = f'bin_edges_sub_length{sub_length}'
hist, bin_edges = np.histogram(data.distances[:,1:],bins=100,density=True)

np.savetxt(fname=f'{distfolder}{hist_filename}_Ns{Ns}.txt',
           X=hist,
)
np.savetxt(fname=f'{distfolder}{bin_edges_filename}_Ns{Ns}.txt',
           X=bin_edges,
)


print(f'this took {(time()-start)/60:.2f} m')