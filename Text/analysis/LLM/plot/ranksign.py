import sys,os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append('../')

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



figsfolder = f'results/sign/figs/ranks/'
os.makedirs(figsfolder,exist_ok=True)
wd = os.environ['WORK']
path0 = "/sacevedo/Data/Text/"

corpus = 'Wikitext'
LLM = 'Pythia'
Lconcat = 150
batch_randomize = 0
layer_ids = [0,24]

signfolder0 = wd + path0 + f'{corpus}/{LLM}/sign/'
if batch_randomize:
  signfolder0 += f'Lconcat{Lconcat}/'

batch_size = 100
Ns_list = np.array([60]) * batch_size
fig,ax = plt.subplots(1)
neighbour = 1
norm_flag = int(sys.argv[1])
for layer_id_aux,layer_id in enumerate(layer_ids):
  for Ns_id,Ns in enumerate(Ns_list):
    resultsfolder = signfolder0 + f'/layer{layer_id}/Ns{Ns}/neighbour{neighbour}/'
    lbl=f'layer {layer_id}'
    R,NR = np.loadtxt(fname=f'{resultsfolder}/ranks.txt',unpack=True)
    R += 1 # lets start the ranks in 1...
    if norm_flag == 1:
      normalization =  np.mean(NR)
    else:
      normalization = 1
    R_values,R_counts = np.unique(R,return_counts=True)
    R_probs = R_counts / np.sum(R_counts)
    ax.plot(R_values / normalization ,R_probs,'o-',label=lbl,color=colors[layer_id_aux])
    mean_R = np.dot(R_probs/normalization,R_values)
    mean_R = np.mean(R)/normalization
    print(f'{Ns=},{mean_R=}')
    ax.vlines(mean_R,min(R_probs),max(R_probs),color=colors[layer_id_aux],linestyles='dashed')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylabel(f'Probability(Rank)')

if normalization ==1:
  ax.set_xlabel('Rank')
else:
  ax.set_xlabel(r'$Rank/ \langle N_{Ranks} \rangle$')
# ax.set_title(f'{d=}')
fig.savefig(fname=figsfolder + f'ranks.pdf')

