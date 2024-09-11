from parameters import *
from utils import *
from dadapy import Data
os.environ["JAX_ENABLE_X64"] = "True"
from dadapy.hamming import Hamming
from time import time

neighbours = [1]
N_batches = 1
Ns = N_batches * batch_size
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
assert layer_id in layer_ids

a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
B,T,E, = a.shape

data = Data(coordinates=a,maxk=np.max(neighbours))
data.compute_distances(maxk=np.max(neighbours))

H = Hamming(coordinates=np.sign(a))
H.compute_distances()
H.distances += H.distances.T # symmetric

for neighbour_id,neighbour in neighbours:
  resultsfolder = signfolder0 + f'/layer{layer_id}/Ns{Ns}/neighbour{neighbour}/'
  os.makedirs(resultsfolder,exist_ok=True)
  R = np.empty(shape=(Ns,))  # ranks in spin space 
  NR = np.empty(shape=(Ns,)) # total number of ranks sampled

  for sample_idx in range(Ns):
    neighbour_idx = data.dist_indices[sample_idx,neighbour]
    D_values, D_counts = np.unique(H.distances[sample_idx,:], return_counts=True)
    assert D_values[0] == 0  # trivial zero
    D_counts[0] -= 1
    if D_counts[0] == 0:
      D_values = D_values[1:]
      D_counts = D_counts[1:]
    R[sample_idx] = np.where(D_values == H.distances[sample_idx,neighbour_idx])[0][0]
    NR[sample_idx] = np.shape(D_values)[0]

    np.savetxt(fname=f'{resultsfolder}hist{sample_idx}.txt',X=np.transpose([D_values,D_counts]))
  np.savetxt(fname=f'{resultsfolder}ranks.txt',X=np.transpose([R,NR]))
