from parameters import *
import matplotlib.pyplot as plt
import sys
sys.path.append('../LLM/')
from utils import *
start = time()
np.set_printoptions(precision=5,suppress=True)

pdffolder = f'results/{corpus}/{LLM}/pdfs/Ntokens{Ntokens}/'
os.makedirs(pdffolder,exist_ok=True)

eps = 1E-7
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'

N_batches = int(sys.argv[7]) # int(os.getenv('N_batches'))
Ns = N_batches * batch_size
print(f'{Ns=}')
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
a = np.unique(a,axis=0)
print(f'{np.mean(a[:,:sub_length,:])=}')


hist_filename = f'histogram_sub_length{sub_length}_layer{layer_id}_Ns{Ns}'
bin_edges_filename = f'bin_edges_sub_length{sub_length}_layer{layer_id}_Ns{Ns}'
hist, bin_edges = np.histogram(a[:,:sub_length,:],bins=100,density=True)
np.savetxt(fname=f'{pdffolder}{hist_filename}_Ns{Ns}.txt',
           X=hist,
)
np.savetxt(fname=f'{pdffolder}{bin_edges_filename}_Ns{Ns}.txt',
           X=bin_edges,
)

print(f'this took {(time()-start)/60. } m')

