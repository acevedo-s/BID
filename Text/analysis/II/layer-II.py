from parameters import *
import matplotlib.pyplot as plt

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


figsfolder = f'results/{corpus}/{LLM}/figs/ranks/'

fig,ax = plt.subplots(1)
sub_lengths = np.array([300]) #np.arange(100,300+1,100,dtype=int)
layer_ids = np.arange(0,24+1,dtype=int)
# layer_ids = [1,14,19,24]
layer_normalize_flags = [0]

II_RS = np.zeros(shape=(len(layer_ids),
                        len(sub_lengths),
                        len(layer_normalize_flags))
)
II_SR = np.zeros(shape=II_RS.shape)

for sub_length_id,sub_length in enumerate(sub_lengths):
  # Ntokens = sub_length - 1
  Ntokens = 0
  for layer_normalize_id,layer_normalize in enumerate(layer_normalize_flags):
    for layer_id_aux,layer_id in enumerate(layer_ids):
      lbl_RS = f'RS;T={sub_length}'
      lbl_SR = f'SR;T={sub_length}'
      if layer_normalize:
        lbl_SR += f' norm'
        lbl_RS += f' norm'
      distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize,Ntokens=Ntokens)
      RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
      RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt')
      II_RS[layer_id_aux,sub_length_id,layer_normalize_id] = 2 * np.mean(RS) / (len(RS) - 1)

      SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
      SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt')
      II_SR[layer_id_aux,sub_length_id,layer_normalize_id] = 2 * np.mean(SR) / (len(SR) - 1)

    ax.plot(layer_ids,II_RS[:,sub_length_id,layer_normalize_id],'o-',label=f'{lbl_RS}')
    ax.plot(layer_ids,II_SR[:,sub_length_id,layer_normalize_id],'s--',label=f'{lbl_SR}')

# print(f'{layer_ids=}')
# print(f'{II_RS[14,0,0]=}')
# ax.hlines(II_RS[14,0,0],layer_ids[0],layer_ids[-1],linestyles='dashed',color='gray')
ax.hlines(0,layer_ids[0],layer_ids[-1],linestyles='dashed',color='gray')
# ax.legend()
# ax.set_yscale('log')
ax.set_ylabel(r'$\Delta$')
ax.set_xlabel(f'layer index')
# ax.set_title(f'{Ntokens=}')

figname = f'{figsfolder}II-layer_dependence.pdf'

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

os.makedirs(figsfolder,exist_ok=True)
fig.savefig(figname,bbox_inches='tight')

