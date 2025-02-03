from parameters import *
from utils import *
from dadapy.hamming import Hamming
import torch
from time import time
torch.set_printoptions(precision=32,sci_mode=False)
sys.path.append('../../')
from paths import *

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'{device=}')

layer_id = int(sys.argv[6])
print(f'{layer_id=}')
assert layer_id in layer_ids

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
sub_length = 10*task_id

print(f'{sub_length=:d}')

# sigmas = load_sigmas(sigmasfolder0,
#                     sublength_cutoff,
#                     layer_id,
#                     sub_length,
#                     )

Ntokens = sub_length
N_batches = 50
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
sigmas = load_activations(N_batches,
                          act_outputfolder,
                          layer_id,
                          sub_length=sub_length, # 0 means every token
                          ).numpy()
Ns0 = sigmas.shape[0]
sigmas = formatting_activations(sigmas,sub_length,Ns0,layer_normalize=0)
sigmas = np.round(np.sign(sigmas),1)
sigmas[np.where(np.isclose(sigmas,0.))] = -1 # just in case

histfolder = makefolder(base='results/BID/hist/',
                        create_folder=True,
                        LLM=LLM,
                        corpus=corpus,
                        layer_id=layer_id,
                        sub_length=sub_length,
                        Nbits=Nbits,
                        batch_randomize=batch_randomize,
                        Ns=N_batches*batch_size,
                        )
H = Hamming(coordinates=sigmas)
start = time()
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              resultsfolder=histfolder,
              )
print(f'{layer_id=:d} took {(time()-start)/60:.1f} min')

# if remove_spins:
#   for t in range(sublength_cutoff):
#     sigmas_filename = get_sigmas_filename(sigmasfolder0,sublength_cutoff,layer_id,t)
#     os.system(f'rm -f {sigmas_filename}')
