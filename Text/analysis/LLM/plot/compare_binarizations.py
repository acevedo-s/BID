import sys,os
import matplotlib.pyplot as plt
import numpy as np
#for fancy plotting
lsize = 20
plt.rcParams['xtick.labelsize']=lsize
plt.rcParams['ytick.labelsize']=lsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = lsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=3)
# markers = ['p','p','o','o','x','x']
markers = ['p','^','s','o']

figsfolder = f'results/'
eps = 1E-7

fig,ax = plt.subplots(1)
l_ids = [0,24]
log_scale = 1
LLM_list = ['Pythia'] #[sys.argv[1]]
corpus = 'Wikitext' #sys.argv[2]
alphamin = 1E-5 #float(sys.argv[3])
alphamax = .7 #np.linspace(.15,.7,15)[-1]
randomize = 0


plot_id = 0 
start = 0
for layer_id in l_ids:
  for Nbits in [1,2]:
    for Lconcat in [None]:
      print(f'{layer_id=}')
      for LLM_id,LLM in enumerate(LLM_list):
        optfolder = f'results/{corpus}/{LLM}/opt/'
        lbl = f'l={layer_id},{Nbits=}'
        if randomize:
            optfolder += f'randomize/'
        if Lconcat != None:
            optfolder += f'Lconcat{Lconcat}/'
            lbl += f' Concat'
        if Nbits > 1:
            optfolder += f'Nbits{Nbits}/'
        lst = '-'
        output_filename = optfolder + f'length_dependence_l_id{layer_id}_alphamax{alphamax:.5f}_alphamin{alphamin:.5f}.txt'
        A = np.loadtxt(output_filename)
        sub_lengths = A[:,0].astype(int)
        sigma_list = A[:,1]
        ax.scatter(sub_lengths[start:],
                    sigma_list[start:],
                    edgecolor='black',
                    marker=markers[plot_id],
                    color=colors[plot_id],
                    label=lbl,
                    zorder=1,
                    )
        ax.plot(sub_lengths[start:],
                sigma_list[start:],
                linestyle=lst,
                color=colors[plot_id],
                zorder=0)
        plot_id += 1
        p,cov = np.polyfit(sub_lengths[start:],sigma_list[start:],1,cov=True)
ax.set_ylabel('BID per bit')
ax.set_xlabel(r'Number of tokens (T)')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend(loc='best',
#           ncol=2,
#           prop={'size':18}
#           )
if log_scale:
  ax.set_yscale('log')
  ax.set_xscale('log')
title = f'{corpus}'
ax.set_title(title)

ax.set_xlim(sub_lengths[0]-5,
            sub_lengths[-1]+10,
            )
figsfolder += f'comparison_binarizations/'
os.makedirs(figsfolder, exist_ok=True)
fig.savefig(figsfolder + f'comparison_{LLM}_{corpus}-BID_alphamax{alphamax:.5f}_alphamin{alphamin:.5f}.pdf',
            bbox_inches='tight')




      

      