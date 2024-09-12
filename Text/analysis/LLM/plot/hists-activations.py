import sys
sys.path.append('LLM/')
from parameters import *
from time import time 
import torch
import matplotlib.pyplot as plt
import numpy as np


#for fancy plotting
rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)

os.makedirs(hist_actfolder,exist_ok=True)
plot_id = 0
N_batches = 70
l_id = int(sys.argv[5])
### LOADING DATA
starting_time = time()
sub_lengths = [10]
fig,ax = plt.subplots()


for sub_lengths_id,sub_length in enumerate(sub_lengths):
  if LLM == 'OPT':
    sub_length += 1
  act_outputfolder = act_outputfolder0 + f'sub_length{sub_length:d}/'
  for batch_id in range(N_batches):
    a_filename = f'{act_outputfolder}b{batch_id}_l{l_id}.pt'
    if batch_id == 0:
      a = torch.load(a_filename,map_location=torch.device('cpu'))
    else:
      a = torch.cat((a,
                    torch.load(a_filename,map_location=torch.device('cpu')))
                    )
  print(f'importing took {(time()-starting_time):.1f}')
  if LLM == 'OPT':
    a = a[:,1:,:] # Every sentence starts with the BOS=2 token
  a_mean = torch.mean(a,axis=0)
  print(f'{a.shape=}')
  (Ns,sub_length,emb_dim) = a.shape # redefining sub_length
  N = emb_dim * sub_length
  lbl = f'l={sub_length}'
  a = torch.flatten(a)
  print(f'{a.shape=}')
  counts, bin_edges = np.histogram(a,bins=100)
  ax.hist(
          bin_edges[:-1],
          bin_edges, 
          weights=counts,
          density=True,
          alpha=.5,
          label=lbl,
          #  color='black',
          )
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.set_title(f'layer:{l_id}')
fname = f'{hist_actfolder}sublength{sub_length}_l{l_id}.pdf'
fig.savefig(fname=fname,bbox_inches='tight')


