from parameters import *
import matplotlib.pyplot as plt
import sys
sys.path.append('../LLM/')
from utils import *
np.set_printoptions(precision=5,suppress=True)

cdffolder = f'results/{corpus}/{LLM}/cdfs/Ntokens{Ntokens}/'
os.makedirs(cdffolder,exist_ok=True)

eps = 1E-7
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'

N_batches = int(os.getenv('N_batches'))
Ns = N_batches * batch_size
print(f'{Ns=}')
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
a = np.unique(a,axis=0)
print(f'{np.mean(a[:,:sub_length,:])=}')

sorted_a, acumulated_prob = np_activations_CDF(a[:,:sub_length,:])
np.savetxt(cdffolder + f'cdf_sub_length{sub_length}_layer{layer_id}_Ns{Ns}.txt',
            np.transpose([sorted_a,acumulated_prob])
)

