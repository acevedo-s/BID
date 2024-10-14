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
actfolder = f'results/{corpus}/{LLM}/peaks/'


layer_normalize_flags = [0]
sample_idx0 = 3
print(f'----------------')
for layer_normalize_id,layer_normalize in enumerate(layer_normalize_flags):
  distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize)
  fig,ax = plt.subplots(1)

  # # LOAD RANKS 
  # RS_ranks_filename = f'ranks_RS_sub_length{sub_length}'
  # RS = np.loadtxt(f'{distfolder}{RS_ranks_filename}.txt').astype(int)
  # print(f'{RS[sample_idx0]=}')
  # # II_RS[layer_id_aux,sub_length_id,layer_normalize_id] = 2 * np.mean(RS) / (len(RS) - 1)

  # SR_ranks_filename = f'ranks_SR_sub_length{sub_length}'
  # SR = np.loadtxt(f'{distfolder}{SR_ranks_filename}.txt').astype(int)
  # print(f'{SR[sample_idx0]=}')
  # # II_SR[layer_id_aux,sub_length_id,layer_normalize_id] = 2 * np.mean(SR) / (len(SR) - 1)
  # print('')

  a_0 = np.loadtxt(actfolder + f'a0{sample_idx0}_layer{layer_id}.txt')
  a_s = np.loadtxt(actfolder + f'a_nns_layer{layer_id}_s0{sample_idx0}.txt')
  a_0 = np.abs(a_0)
  a_s = np.abs(a_s)

  # print(f'{np.max(a_0)=}')
  # print(f'{np.max(a_s)=}')

  a_0 = np.sort(a_0)
  a_s = np.sort(a_s)
  ax.plot(a_0[:])
  ax.plot(a_s[:])
  # ax.legend()
  ax.set_yscale('log')
  ax.set_xscale('log')
  # ax.set_ylabel(r'$\Delta$')
  # ax.set_xlabel(f'layer index')
  # ax.set_title(f'{}')
  figname = f'{figsfolder}act{layer_id}.pdf'

  # box = ax.get_position()
  # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  ax.grid()
  # h = 10**3
  # ax.hlines(h,0,len(a_0),color='black')
  ax.set_ylim(10**(-5),10**(4))
  fig.savefig(figname,bbox_inches='tight')

