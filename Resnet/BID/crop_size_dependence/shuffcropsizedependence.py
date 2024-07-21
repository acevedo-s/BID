import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *
from R.relative_depth import * 

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
# print(plt.rcParams.keys())
np.set_printoptions(precision=7)
# plt.xticks(rotation=90)
markers = ['p','^','s','o','>','<']

# crop_sizes = [122,244]
# crop_step = 28
# crop_sizes = np.array([crop_step*i for i in range(1,8+1)])
crop_step = 56
crop_sizes = np.array(
  # [28] +
  [crop_step*i for i in range(1,4+1)]
)
figsfolder = f'results/figs/'
IDfolder = f'results/ID/shuffled/'
os.makedirs(IDfolder,exist_ok=True)
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

delta = 5E-4
Nsteps = int(5E5)
seed = 100

model_id = 0
model_name,W_model = model_list[model_id]

layer_names = layers_dict[model_name]
l1 = 0
l2 = -1 #l1 + 2

# this discards flatten:
layer_names = layer_names[l1:l2]
relative_depth_dict[model_name] = relative_depth_dict[model_name][l1:l2]

#selected layers:
# indices = range(0,8)
indices = [0,1,2,3,4,5,6,7]
# indices = [6]
layer_names = [layer_names[idx] for idx in indices]
relative_depth_dict[model_name] = [relative_depth_dict[model_name][idx] for idx in indices]

# ED_list = np.zeros(shape=(len(layer_names)),
#                           dtype=int
#                    )
nc = 7
class_list = list(class_dict.keys())
# class_list = class_list[nc:nc+1]
class_list = class_list[:nc]

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

figKL,axKL = plt.subplots(1)
figs,axs = plt.subplots(1,
                        # figsize=(10,5),
                        )
figa,axa = plt.subplots(1)

alphamin = 0.01
sigma_list = np.zeros(shape=(len(class_list),
                             len(layer_names),
                             len(crop_sizes),
                             )
                      )
alpha_list = np.zeros(shape=sigma_list.shape)
KL_list = np.zeros(shape=sigma_list.shape)
normalize = int(sys.argv[1])
log_scale = int(sys.argv[2])

for crop_id,crop_size in enumerate(crop_sizes):
  print(f'{crop_size=}')
  histfolder = f'../../distances/results/{model_name}/hist/crop_size{crop_size}/'
  optfolder0 = f'../optimize/results/opt/{model_name}/crop_size{crop_size}/'
  for r_id in range(len(class_list)):
    for layer_id,layer_name in enumerate(layer_names):
      if normalize:
        norm = crop_size**2
      else:
        norm = 1
      # print(f'{layer_name},{N=}')
      optfolder = optfolder0 + f'shuffled/{r_id}/{layer_name}/alphamin{alphamin:.5f}/'
      optfolder += f'delta{delta}/seed{seed}/'
      optfile = optfolder + 'opt.txt'
      H = Hamming(crossed_distances=crossed_distances)
      H.D_histogram(
        r_id=r_id,
        resultsfolder=histfolder+f'shuffled/{layer_name}/',
        )
      H.set_r_quantile(alphamin)
      rmin = H.r
      idmin = H.r_idx
      H.r = None
      H.r_idx = None
      try:
        A = np.loadtxt(optfile,delimiter=',')
        fail = 0
      except:
        A = np.array([np.nan for i in range(4)])
        fail = 1
      if len(A.shape)==1:
        A = np.vstack([A,A])
      rmaxs = A[:,0]
      sigmas = A[:,1]
      alphas = A[:,2]
      logKLs = A[:,3]
      minKL_id = np.where(np.isclose(logKLs,np.nanmin(logKLs)))[0] # this is a list
      if len(minKL_id)==0:
        minKL_id = 0
      else:
        minKL_id = minKL_id[0]
      sigma_list[r_id,layer_id,crop_id] = sigmas[minKL_id] / norm
      alpha_list[r_id,layer_id,crop_id] = alphas[minKL_id]
      KL_list[r_id,layer_id,crop_id] = logKLs[minKL_id]
      if fail:
        print(f'{r_id},{layer_name}: {sigmas[minKL_id]=}')

mu = np.nanmean(sigma_list,axis=0)
std = np.nanstd(sigma_list,axis=0)
mu_a = np.nanmean(alpha_list,axis=0)
std_a = np.nanstd(alpha_list,axis=0)
mu_KL = np.nanmean(KL_list,axis=0)
std_KL = np.nanstd(KL_list,axis=0)
# axs.errorbar(relative_depth_dict[model_name],
#              mu,
#              std,
#              capsize=12,
#             #  'o-',
#             label=f'ID',
#             )
plot_id = 0
for layer_id,layer_name in enumerate(layer_names):
  lbl = r'$l/L_R=$' + f'{relative_depth_dict[model_name][layer_id]:.2f}'
  axs.scatter(crop_sizes**2,
            mu[layer_id,:],
            marker=markers[plot_id%len(markers)],
            edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            )
  np.savetxt(fname=f'{IDfolder}/{layer_name}',
             X=np.transpose([crop_sizes**2,
                             mu[layer_id,:],
                             std[layer_id,:]
                             ]
                            )
            )
  axs.fill_between(crop_sizes**2,
              mu[layer_id,:]-std[layer_id,:],
              mu[layer_id,:]+std[layer_id,:],
              color=colors[plot_id%len(colors)],
              alpha=.5,
              zorder=0,
              )
  plot_id += 1
  lbl = r'$l/L=$'+f'{relative_depth_dict[model_name][layer_id]:.2f}'
  axa.scatter(crop_sizes**2,
            mu_a[layer_id,:],
            marker=markers[plot_id%len(markers)],
            # edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            )
  axKL.scatter(crop_sizes**2,
            mu_KL[layer_id,:],
            marker=markers[plot_id%len(markers)],
            edgecolor='black',
            # label=f'{layer_name}',
            label=lbl,
            color=colors[plot_id%len(colors)],
            )
# axED = axs.twinx()
# axED.set_ylabel('N')
# axED.plot(crop_sizes,
#         ED_list,
#         color='black',
#         linestyle='dashed',
#         label='N',
#         )

if normalize:
  axs.set_ylabel(r'$BID/N_c$')
else:
  axs.set_ylabel(r'BID')
axs.set_xlabel(r'$N_c$')

# axED.ticklabel_format(style='sci',
#                     axis='y', 
#                     scilimits=(5,5), 
#                     useMathText=True
#                     )
# axED.legend(loc='upper right')
if log_scale:
  axs.set_yscale('log')
  axs.set_xscale('log')
# else:
#   axs.ticklabel_format(style='sci',
#                     axis='y', 
#                     scilimits=(5,5), 
#                     useMathText=True
#                     )

axa.set_ylabel(r'$\alpha$')
axa.set_xlabel(f'crop size')
box = axa.get_position()
axa.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axa.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figa.savefig(figsfolder + 'shuff_alpha_crop.pdf',bbox_inches='tight')

# axs.legend(loc='upper left',
#            prop={'size':15},
#            ncol=1)
# axs.set_xlim(0,230)
# axs.legend(loc='upper center', 
#           bbox_to_anchor=(0.5, 1.15),
#           ncol=len(crop_sizes), 
#           # ncol=3,
#           fancybox=True, 
#           shadow=True,
#           prop={'size':15},
#           )
box = axs.get_position()
axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
figs.savefig(figsfolder + f'shuff_sigma_crop_n{normalize}_log_{log_scale}.pdf',
            bbox_inches='tight'
             )

# axs.legend(loc='center')
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figKL.savefig(figsfolder + 'shuff_KL_crop.pdf',bbox_inches='tight')
