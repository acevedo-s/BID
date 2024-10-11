from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *
from time import time 

start = time()
neighbour = 1
distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)

r_idcs_filename = f'r_dist_indices_sub_length{sub_length}'
real_dist_indices = np.loadtxt(f'{distfolder}{r_idcs_filename}.txt').astype(int)

s_dists_filename = f's_dists_sub_length{sub_length}'
spin_distances = np.loadtxt(f'{distfolder}{s_dists_filename}.txt').astype(int)

SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
ranks_spin_to_real(real_dist_indices,
                  spin_distances,
                  neighbour,
                  distfolder,
                  filename=SR_ranks_filename,
                  )

print(f'this took {(time()-start)/60:.1f} min')
