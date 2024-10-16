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

figsfolder = f'results/{corpus}/{LLM}/figs/peaks/Ntokens{Ntokens}/'
actfolder = f'results/{corpus}/{LLM}/peaks/Ntokens{Ntokens}/'

layer_normalize_flags = [0]
sample_idx0 = int(sys.argv[7])
fig,ax = plt.subplots(1)

for layer_normalize_id,layer_normalize in enumerate(layer_normalize_flags):
  distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize,Ntokens=Ntokens)

  a_0 = np.loadtxt(actfolder + f'a0{sample_idx0}_layer{layer_id}.txt')
  a_s = np.loadtxt(actfolder + f'a_nns_layer{layer_id}_s0{sample_idx0}.txt')
  # a_0 = np.abs(a_0)
  # a_s = np.abs(a_s)
  
  # a_0 = np.sort(a_0)
  # a_s = np.sort(a_s)
  ax.plot(a_0[:],'x')
  ax.plot(a_s[:],'x')
  # ax.legend()
  # ax.set_yscale('log')
  # ax.set_xscale('log')
  # ax.set_ylim(10**(-5),10**(4))

  ax.set_ylabel(r'$a_j$')
  ax.set_xlabel(f'j')
  ax.set_title(f'{layer_id=}')
  figname = f'{figsfolder}act{layer_id}_sample{sample_idx0}.png'

  # box = ax.get_position()
  # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.grid()
  # h = 10**3
  
  fig.savefig(figname,bbox_inches='tight')

