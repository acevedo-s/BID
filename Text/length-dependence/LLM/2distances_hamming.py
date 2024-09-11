from parameters import *
from utils import *
from time import time
from dadapy import Hamming
t = int(os.getenv('t'))
N_batches = 60
Ns = N_batches * batch_size
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
tau = int(sys.argv[7])
print(f'{tau=}')
assert layer_id in layer_ids

print(f'loading activations')
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
B,T,E, = a.shape

sigmas = np.sign(a)
sigmas[np.asarray(sigmas == 0).nonzero()] = -1
sigmas = np.concatenate((sigmas[:,t,:],
                         sigmas[:,t+tau,:]),
                         axis=1)
sigmas = np.unique(sigmas,axis=0)
print(f'{sigmas.shape=}')

H = Hamming(coordinates=sigmas)
start = time()
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              resultsfolder=histfolder+f'2distances/layer{layer_id}/tau{tau}/')

print(f'this took {(time()-start)/60:.1f} min')
