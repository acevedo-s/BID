from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *
from time import time 

start = time()
neighbour = 1
distfolder = get_distfolder(corpus,
                            LLM,
                            layer_id,
                            layer_normalize,
                            Ntokens=Ntokens,
)

r_idcs_filename = f'r_dist_indices_sub_length{sub_length}'
real_dist_indices = np.loadtxt(f'{distfolder}{r_idcs_filename}.txt').astype(int)

s_dists_filename = f's_dists_sub_length{sub_length}'
spin_distances = np.loadtxt(f'{distfolder}{s_dists_filename}.txt').astype(int)

RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
ranks_real_to_spin(real_dist_indices,
                  spin_distances,
                  neighbour,
                  distfolder,
                  filename=RS_ranks_filename,
                  )

print(f'this took {(time()-start)/60:.1f} min')
