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
crossed_distances = 0

fig,axs = plt.subplots(1)
l_ids = [24]
log_scale = 1
LLM = sys.argv[1]
corpus = sys.argv[2]
alphamax = float(sys.argv[3])
randomize = int(sys.argv[4])

start = 0

plot_id = 0 
for layer_id in l_ids: 
  if layer_id == 24 and randomize == 0: plot_id +=1
  optfolder = f'results/{corpus}/{LLM}/opt/'
  if randomize:
    lbl = r'$\tilde{l}=$' + f'{layer_id}'
    optfolder += f'randomize/'
    lst = '--'
  else:
    lbl = f' l={layer_id}'
    lst = '-'
  output_filename = optfolder + f'length_dependence_l_id{layer_id}_alphamax{alphamax:.5f}.txt'
  A = np.loadtxt(output_filename)
  sub_lengths = A[:,0].astype(int)
  sigma_list = A[:,1]
  axs.scatter(sub_lengths[start:],
              sigma_list[start:],
              edgecolor='black',
              marker=markers[plot_id],
              color=colors[plot_id],
              label=lbl,
              zorder=1,
              )
  axs.plot(sub_lengths[start:],
          sigma_list[start:],
          linestyle=lst,
          color=colors[plot_id],
          zorder=0)
  plot_id += 1

if randomize:
  plot_id = 3
  axst = axs.twinx()
  layer_id = 24
  optfolder = f'results/{corpus}/{LLM}/opt/'
  lbl = f' l={layer_id}'
  output_filename = optfolder + f'length_dependence_l_id{layer_id}_alphamax{alphamax:.5f}.txt'
  A = np.loadtxt(output_filename)
  sub_lengths = A[:,0].astype(int)
  sigma_list = A[:,1]
  axst.scatter(sub_lengths[start:],
              sigma_list[start:],
              edgecolor='black',
              marker=markers[plot_id],
              color=colors[plot_id],
              label=lbl,
              zorder=1,
              )
  axst.plot(sub_lengths[start:],
          sigma_list[start:],
          '-',
          color=colors[plot_id],
          zorder=0)
  axst.tick_params(axis='both', 
                   which='both', 
                   bottom=False, 
                   top=False, 
                   left=False, 
                   right=False,
                   labelright=False, 
                   labelbottom=False)
  
    
if False:
  corpus = 'OWebtext'
  l_ids = [24]
  log_scale = 1
  randomizing_flags = [0]
  for layer_id in l_ids: 
    for randomize in randomizing_flags:
      n = np.round(float(sys.argv[1]),decimals=1)
      # if randomize == 1 and layer_id == 0:
      #   n = 3
      optfolder = f'results/{corpus}/opt/'
      lbl = corpus + f' l={layer_id}'
      if randomize:
        optfolder += f'randomize/'
        output_filename = optfolder + f'length_dependence_l_id{layer_id}_n{n:.2f}_randomize.txt'
        lbl += f' random'
      else:
        output_filename = optfolder + f'length_dependence_l_id{layer_id}_n{n:.2f}_reg.txt'
      figsfolder = f'results/'
      A = np.loadtxt(output_filename)
      sub_lengths = A[:,0].astype(int)
      sigma_list = A[:,1]
      # alpha_list = A[:,2]
      # KL_list = A[:,3]
      # rmax_list = A[:,4].astype(int)
      mark = 's'
      axs.plot(sub_lengths,
              sigma_list,
              marker=mark,
              label=lbl)
    
if False:
  corpus = 'Tinystories'
  l_ids = [24]
  sigma_list = []
  randomizing_flags = [0,1]
  for layer_id in l_ids: 
    for randomize in randomizing_flags:
      optfolder = f'results/{corpus}/opt/'
      lbl = corpus + f' l={layer_id}'
      output_filename = optfolder + f'length_dependence_l_id{layer_id}_n{n}_reg.txt'
      if randomize:
        optfolder += f'randomize/'
        output_filename = optfolder + f'length_dependence_l_id{layer_id}_n{n}_randomize.txt'
        lbl += f' random'
      figsfolder = f'results/'
      A = np.loadtxt(output_filename)
      sub_lengths = A[:,0].astype(int)
      sigma_list = A[:,1]
      # alpha_list = A[:,2]
      # KL_list = A[:,3]
      # rmax_list = A[:,4].astype(int)
      mark = '^'
      axs.plot(sub_lengths,
              sigma_list,
              marker=mark,
              label=lbl)
        

if False:
  ###Random tokens: 
  corpus = 'Randomtokens'
  for layer_id in [24]:
    optfolder = f'results/{corpus}/opt/'
    lbl = corpus + f' l={layer_id}'
    output_filename = optfolder + f'length_dependence_l_id{layer_id}_n{n}.txt'
    figsfolder = f'results/'
    A = np.loadtxt(output_filename)
    sub_lengths = A[:,0].astype(int)
    sigma_list = A[:,1]
    alpha_list = A[:,2]
    KL_list = A[:,3]
    rmax_list = A[:,4].astype(int)
    axs.plot(sub_lengths,
            sigma_list,
            'o-',
            label=lbl,
            color='black',
            )

axs.set_ylabel('BID per bit')
# axs.set_xlabel(f'number of tokens')
axs.set_xlabel(r'Number of tokens (T)')
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# axs.legend(loc='best')
if randomize:
  axst.legend(loc='lower left')

if log_scale:
  axs.set_yscale('log')
  axs.set_xscale('log')
  if randomize:
    axst.set_yscale('log')
    axst.set_xscale('log')

if randomize:
  ymin = axst.get_ylim()[0] * .85
  ymax = axs.get_ylim()[1]
  axs.set_ylim(ymin,ymax)
  axst.set_ylim(ymin,ymax)

print(f'{alphamax=}')
title = f'{corpus}'
if randomize:
  title += ' random'
axs.set_title(title)

# axs.grid()
fig.savefig(figsfolder + f'{LLM}-{corpus}-BID_rand{randomize}_alphamax{alphamax:.2f}.pdf',bbox_inches='tight')


      

      