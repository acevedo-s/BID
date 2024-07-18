from parameters import *
from utils import *
from dadapy import Hamming
import torch
from time import time
torch.set_printoptions(precision=32,sci_mode=False)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'{device=}')

layer_id = int(sys.argv[6])
print(f'{layer_id=}')
assert layer_id in layer_ids

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
sub_length = 10*task_id

print(f'{sub_length=:d}')
# sigmasfolder = f'{sigmasfolder0}sub_length{sublength_cutoff:d}/'
# sigmas_filename = f'{sigmasfolder}sigmas_l{layer_id}.pt'
# sigmas = torch.Tensor.numpy(torch.load(sigmas_filename,
#                                         map_location=torch.device('cpu')
#                                         )
#                             )

# ### SUBSENTENCE THRESHOLDING
# sigmas = sigmas[:,:sub_length,:]
# print(f'{sigmas.shape=}')

# ### REMOVING REPETITIONS (must be done again for each sub sentence length...)
# sigmas = np.unique(sigmas,axis=0) # this reorders the sentences
# print(f'after removing repeated strings:{sigmas.shape=}')

# print(f'{sigmas.shape=}')
# (Ns,sub_length,emb_dim) = sigmas.shape
# N = emb_dim * sub_length
# sigmas = np.reshape(sigmas,(Ns,N))

sigmas = load_sigmas(sigmasfolder0,
                    sublength_cutoff,
                    layer_id,
                    sub_length,
                    )
H = Hamming(coordinates=sigmas)
start = time()
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              k=sub_length,
              t=layer_id,
              resultsfolder=histfolder)
print(f'{layer_id=:d} took {(time()-start)/60:.1f} min')

if remove_spins:
  for t in range(sublength_cutoff):
    sigmas_filename = get_sigmas_filename(sigmasfolder0,sublength_cutoff,layer_id,t)
    os.system(f'rm -f {sigmas_filename}')
