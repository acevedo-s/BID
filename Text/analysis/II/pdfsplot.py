import matplotlib.pyplot as plt
import numpy as np
import os,sys
# import sys
# sys.path.append('../LLM/')
# np.set_printoptions(precision=5,suppress=True)

# rcpsize = 20
# plt.rcParams['xtick.labelsize']= rcpsize
# plt.rcParams['ytick.labelsize']=rcpsize
# plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.family'] = 'STIXGeneral'
# plt.rcParams['font.size'] = rcpsize
# plt.rcParams.update({'figure.autolayout': True})
# #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
# colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
# from cycler import cycler
# plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# # print(plt.rcParams.keys())
# #np.set_printoptions(precision=None)
# markers = ['p','o','h','^','s']
# plot_id = 0

LLM = 'OPT'
corpus = 'Wikitext' # 'OWebtext'
Ntokens = 0

figsfolder = f'results/{corpus}/{LLM}/figs/pdfs/Ntokens{Ntokens}/'
os.makedirs(figsfolder,exist_ok=True)
pdffolder = f'results/{corpus}/{LLM}/pdfs/Ntokens{Ntokens}/'


N_batches = 1
batch_size = 100
Ns = N_batches * batch_size
sub_lengths = np.array([20])
layer_ids = [0,18,24]
fig,ax = plt.subplots(1)

for sub_length_id,sub_length in enumerate(sub_lengths):
  for layer_id_aux,layer_id in enumerate(layer_ids):
    hist_filename = f'histogram_sub_length{sub_length}_layer{layer_id}_Ns{Ns}'
    bin_edges_filename = f'bin_edges_sub_length{sub_length}_layer{layer_id}_Ns{Ns}'
    hist = np.loadtxt(fname=f'{pdffolder}{hist_filename}.txt')
    bin_edges = np.loadtxt(fname=f'{pdffolder}{bin_edges_filename}.txt')
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    lbl = f'l={layer_id};T={sub_length}'
    ax.plot(bin_centers,hist,'x',label=lbl)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.vlines(0,0,1,color='black')
    figname = f'{figsfolder}act_pdf_layer{layer_ids}_Ns{Ns}_sub_lengths{sub_lengths}.png'
    ax.legend()
    ax.set_ylabel(r'$pdf(a)$')
    ax.set_xlabel(f'a')
fig.savefig(figname,bbox_inches='tight')


# plot_id += 1

  # sorted_a, acumulated_prob = np_activations_CDF(-a[:,:sub_length,:])
  # ax.scatter(sorted_a,acumulated_prob, facecolors='none', edgecolors=colors[plot_id])
  # plot_id += 1
  # distfolder = get_distfolder(corpus,LLM,layer_id,layer_normalize,Ntokens=Ntokens)


#   ax.set_title(f'{layer_id=}')
#   ax.grid()




# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  

