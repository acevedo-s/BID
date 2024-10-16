from parameters import *
import matplotlib.pyplot as plt
import sys
sys.path.append('../LLM/')
from utils import *
np.set_printoptions(precision=5,suppress=True)

figsfolder = f'results/{corpus}/{LLM}/figs/peaks/Ntokens{Ntokens}/'
os.makedirs(figsfolder,exist_ok=True)
actfolder = f'results/{corpus}/{LLM}/peaks/Ntokens{Ntokens}/'
os.makedirs(actfolder,exist_ok=True)

sample_idx0 = int(sys.argv[7])
print(f'{sample_idx0=}')
# N_batches = int(os.getenv('N_batches'))

distfolder = get_distfolder(corpus,
                            LLM,
                            layer_id,
                            layer_normalize,
                            Ntokens=Ntokens,
                            )

# LOAD ACTIVATIONS
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
a = formatting_activations(a,sub_length,N_batches * batch_size,layer_normalize)

print(f'{a[sample_idx0,:]=}')
np.savetxt(actfolder + f'a0{sample_idx0}_layer{layer_id}.txt',
            np.transpose(a[sample_idx0,:]))

# WHO IS THE FIRST NEIGHBOUR IN SPIN SPACE? 
s_dists_filename = f's_dists_sub_length{sub_length}'
spin_distances = np.loadtxt(f'{distfolder}{s_dists_filename}.txt').astype(int)
min_dist = np.min(np.delete(spin_distances[sample_idx0,:],sample_idx0)) # minimum hamming distance from sample, excluding the self-distance...
neighbour_idcs = np.where(spin_distances[sample_idx0,:] == min_dist)[0] # list of indeces that share the minimum distance in spin spaces (variable size)
print(f'{min_dist=}')
print(f'{neighbour_idcs=}')

a_nns = []
for neighbour_idx in neighbour_idcs:
  a_nns.append(a[neighbour_idx,:])
  print(a[neighbour_idx,:])

a_nns = np.array(a_nns)
np.savetxt(actfolder + f'a_nns_layer{layer_id}_s0{sample_idx0}.txt',
            np.transpose(a_nns))