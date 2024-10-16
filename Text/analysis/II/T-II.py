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

### for legend outside plot:
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

figsfolder = f'results/{corpus}/{LLM}/figs/ranks/'
os.makedirs(figsfolder,exist_ok=True)

fig,ax = plt.subplots(1)
sub_lengths = np.arange(20,300+1,20,dtype=int)

II_RS = np.zeros(shape=(len(sub_lengths)))
II_SR = np.zeros(shape=II_RS.shape)

for sub_length_id,sub_length in enumerate(sub_lengths):
  distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize,Ntokens=Ntokens)
  RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
  RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt')
  II_RS[sub_length_id] = 2 * np.mean(RS) / (len(RS) - 1)

  SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
  SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt')
  II_SR[sub_length_id] = 2 * np.mean(SR) / (len(SR) - 1)

ax.plot(sub_lengths,II_RS,'o-',label=f'RS')
ax.plot(sub_lengths,II_SR,'o-',label=f'SR')
ax.legend()
ax.set_yscale('log')
ax.set_ylabel(r'$\Delta$')
ax.set_xlabel(f'Number of tokens T')
ax.set_title(f'{Ns=};{layer_id=}')
figname = f'{figsfolder}II_layer{layer_id}.pdf'
fig.savefig(figname,bbox_inches='tight')

