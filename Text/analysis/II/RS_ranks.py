from parameters import *
import sys
sys.path.append('../LLM/')
from utils import *

neighbour = 1
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