import matplotlib.pyplot as plt
import numpy as np
import os,sys
from parameters import get_distfolder

LLM = 'OPT'
corpus = 'OWebtext'
Ntokens = 0

figsfolder = f'results/{corpus}/{LLM}/figs/hists/Ntokens{Ntokens}/'
os.makedirs(figsfolder,exist_ok=True)


N_batches_list = [50]
batch_size = 100
sub_lengths = np.array([20,200])
layer_ids = range(25)


for layer_id_aux,layer_id in enumerate(layer_ids):
  fig,ax = plt.subplots(1)
  for sub_length_id,sub_length in enumerate(sub_lengths):
    for N_batches in N_batches_list:
      Ns = N_batches * batch_size
      distfolder = get_distfolder(corpus,
                                  LLM,
                                  layer_id,
                                  layer_normalize=0,
                                  Ntokens=Ntokens,
      )

      lbl = f'l={layer_id};T={sub_length}'
      hist_filename = f'histogram_sub_length{sub_length}_Ns{Ns}'
      bin_edges_filename = f'bin_edges_sub_length{sub_length}_Ns{Ns}'
      counts = np.loadtxt(fname=f'{distfolder}{hist_filename}.txt')
      bin_edges = np.loadtxt(fname=f'{distfolder}{bin_edges_filename}.txt')
      bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
      ax.plot(bin_centers,counts,'-x',label=lbl)
      figname = f'{figsfolder}dists_layer{layer_id}_Ns{Ns}_sub_lengths{sub_lengths}.png'
      ax.legend()
      ax.set_yscale('log')    
      ax.set_ylabel(r'$p(r)$')
      ax.set_xlabel(f'r')
    fig.savefig(figname,bbox_inches='tight')
    plt.close()

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


