from parameters import *
from utils import *
import torch
from time import time
torch.set_printoptions(precision=32,sci_mode=False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'{device=}')

t = int(os.getenv('t'))
N_batches = 50
Ns = N_batches * batch_size
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
layer_id = int(sys.argv[6])
print(f'{layer_id=}')
tau = int(sys.argv[7])
print(f'{tau=}')
assert layer_id in layer_ids
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
B,T,E, = a.shape

angles = np.zeros(shape=(Ns))
for sample_idx in range(Ns):
  angles[sample_idx] = np.dot(a[sample_idx,t,:],a[sample_idx,t+tau,:])
  angles[sample_idx] /= np.linalg.norm(a[sample_idx,t,:]) 
  angles[sample_idx] /= np.linalg.norm(a[sample_idx,t+tau,:]) 
print(f'{angles.shape=}')
angles_filename = get_angles_filename(anglesfolder0,
                                      t,
                                      tau,
                                      layer_id)
np.savetxt(fname=angles_filename,
           X=angles)



