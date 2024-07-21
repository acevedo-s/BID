import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from dadapy import Hamming
import os 
import numpy as np
from R.models import *
from R.data import *


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
# markers = ['p','p','o','o','x','x']
markers = ['s']

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
filesfolder = f'../../results/files/'

crop_size = int(sys.argv[1])
model_id = 0
model_name,W_model = model_list[model_id]
histfolder = f'../../distances/results/hist/'#/{model_name}/'
optfolder = f'../optimize/results/opt/{model_name}/crop_size{crop_size}/'
class_list = list(class_dict.keys())[:1]
layer_names = layers_dict[model_name]

l1 = 0
l2 = l1+1
layer_names = layer_names[l1:l2]

eps = 1E-7
metric = 'hamming'
crossed_distances = 0

figKL,axKL = plt.subplots(1)
figs,axs = plt.subplots(1)
figa,axa = plt.subplots(1)

alphamin = 0.01

for key_id,key in enumerate(class_list):
  print(f'{key}({class_dict[key]}):')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  Ns = len(files)
  for layer_id,layer_name in enumerate(layer_names):
    EDfolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/{layer_name}/'
    EDfile = EDfolder + f'a_flatten_shape.txt'
    Ns_act,N = np.genfromtxt(EDfile,
                            dtype='str',
                            unpack=True).astype(int)
    print(f'{layer_name},{N=}')
    optfile = optfolder + f'{key}/{layer_name}/alphamin{alphamin:.5f}.txt'
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
      Ns=Ns,
      resultsfolder=histfolder+f'{key}/{layer_name}/',
      )
    H.set_r_quantile(alphamin)
    rmin = H.r
    idmin = H.r_idx
    H.r = None
    H.r_idx = None
    A = np.loadtxt(optfile,delimiter=',')
    rmaxs = A[:,0]
    sigmas = A[:,1]
    alphas = A[:,2]
    logKLs = A[:,3]
    # minKL_id = np.where(np.isclose(logKLs,min(logKLs)))[0][0]
    axs.plot(rmaxs/N,sigmas/N,'o-',
             label=f'{layer_name}'
            )
    axKL.plot(rmaxs/N,logKLs,
             label=f'{layer_name}'
            )
    axa.plot(rmaxs/N,alphas,
            label=f'{layer_name}'
            )
    #     KL_list.append(Op.KL)
    #     sigma_list.append(Op.sigma)
    #     alpha_list.append(Op.alpha)
    #     rmaxs.append(rmax)
    #     if export_output_flag:
    #       print(f'{rmax:.3f},{Op.sigma:.8f},{Op.alpha:8f},{np.log(Op.KL):.8f}',
    #             file=open(optfile, 'a'))
    #   rmaxs = np.array(rmaxs)
    #   sigma_list = np.array(sigma_list)
    #   alpha_list = np.array(alpha_list)
    #   axKL.plot(rmaxs/N,np.log(np.array(KL_list)),'o-',label=f'{seed=}')
    #   axs.plot(rmaxs/N,sigma_list/N,'o-')
    #   axa.plot(rmaxs/N,alpha_list,'o-')

    # # axs.hlines(s[n_id],
    # #            axs.get_xlim()[0],
    # #            axs.get_xlim()[1],
    # #            linestyles='dashed',
    # #            color='black',
    # #            label=f's_exact',
    # #            )

axKL.set_ylabel('ln(KL)')
# axKL.set_xlabel(r'$\alpha_r$')
axKL.set_xlabel(r'$r^*/N$')
box = axKL.get_position()
axKL.set_position([box.x0, box.y0, box.width * 0.8, box.height])
axKL.legend(loc='center left', bbox_to_anchor=(1, 0.5))
figKL.savefig(figsfolder + 'KL.png',bbox_inches='tight')

axs.set_ylabel(r'$\sigma$/N')
# axs.set_xlabel(r'$\alpha_r$')
axs.set_xlabel(r'$r^*/N$')
axs.legend()
figs.savefig(figsfolder + 'sigma.png')

axa.set_ylabel(r'$\alpha$')
# axa.set_xlabel(r'$\alpha_r$')
axa.set_xlabel(r'$r^*/N$')
figa.savefig(figsfolder + 'alpha.png')

      