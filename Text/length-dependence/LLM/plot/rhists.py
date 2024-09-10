import sys,os
sys.path.append('LLM/')
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from dadapy import Hamming


#for fancy plotting
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
np.set_printoptions(precision=7)
markers = ['s']

figsfolder = f'results/{corpus}/{LLM}/figs/r-hists/'
os.makedirs(figsfolder,exist_ok=True)


# l_id = int(sys.argv[4])
# sub_lengths = np.array([i * 10 + 1 for i in range(1,7+1,2)])
# sub_lengths = np.array([i * 10 for i in [30]])
sub_lengths = [20,100,200,300]
if LLM == 'OPT':
  stupid_correction = 1
else:
  stupid_correction = 0

eps = 1E-7
crossed_distances = 0
metric = 'hamming'

regularize = 1
alphamin = 0
l_ids = [0,5,12,17,23,24]

for sub_length_id,sub_length in enumerate(sub_lengths):
  figh,axh = plt.subplots(1)
  for l_id in l_ids:
    # if l_id == 24 and LLM == 'OPT':
    #   emb_dim = 512
    # else:
    #   emb_dim = 1024
    # if Ntokens != 1:
    #   N = (sub_length) * emb_dim 
    # else:
    #   N = emb_dim
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
              k=sub_length+stupid_correction,
              t=l_id,
              resultsfolder=histfolder
    )
    if alphamin > 0:
      H.set_r_quantile(alphamin)
      rmin = H.r
      idmin = H.r_idx
      H.r = None
      H.r_idx = None
    else:
      idmin = 0
    axh.plot(H.D_values[idmin:],
            H.D_probs[idmin:],
            'x',
          #  color='black',
            label=f'layer {l_id}',
            zorder=0,
            )
  axh.set_title(f'length={sub_length}')
  axh.set_xlabel(r'$r$')
  axh.set_yscale('log')
  box = axh.get_position()
  axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  figh.savefig(figsfolder + f'hists_rand{randomize}_sublength_{sub_length}.png',
              bbox_inches='tight')      