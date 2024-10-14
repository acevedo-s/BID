from parameters import *
import matplotlib.pyplot as plt
import sys
sys.path.append('../LLM/')
from utils import *
np.set_printoptions(precision=5,suppress=True)


rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)
markers = ['p','o','h','^','s']
plot_id = 0


figsfolder = f'results/{corpus}/{LLM}/figs/peaks/'
os.makedirs(figsfolder,exist_ok=True)
actfolder = f'results/{corpus}/{LLM}/peaks/'
os.makedirs(actfolder,exist_ok=True)

sample_idx0 = int(os.getenv('sample_idx0'))
print(f'{sample_idx0=}')
print(f'----------------')
distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)
# LOAD ACTIVATIONS
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
a = load_activations(N_batches,
                    act_outputfolder,
                    layer_id,
                    LLM,
                    Ntokens).numpy()
B,T,E = np.shape(a)

a = np.reshape(a,newshape=(B,T*E))
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
fig,ax = plt.subplots(1)
D_values, D_counts = np.unique(spin_distances[sample_idx0,:],return_counts=True)
assert D_values[0] == 0  # trivial zero
D_counts[0] -= 1
if D_counts[0] == 0:
  D_values = D_values[1:]
  D_counts = D_counts[1:]
ax.plot(D_values,D_counts)
fig.savefig(figsfolder + f'local_hist{sample_idx0}.pdf')

a_nns = []
for neighbour_idx in neighbour_idcs:
  a_nns.append(a[neighbour_idx,:])
  print(a[neighbour_idx,:])

a_nns = np.array(a_nns)
np.savetxt(actfolder + f'a_nns_layer{layer_id}_s0{sample_idx0}.txt',
            np.transpose(a_nns))