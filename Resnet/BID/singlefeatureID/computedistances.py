import matplotlib.pyplot as plt
import sys,glob
sys.path.append('../../')
from R import load_features_layer
from R.models import *
from R.data import *
from dadapy import Hamming
import os 
import numpy as np

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
np.set_printoptions(precision=10)


figsfolder = f'results/figs/'

model_id = 0
model_name,W_model = model_list[model_id]
distfolder0 = f'results/{model_name}/'
os.makedirs(distfolder0, exist_ok=True)

print(f'{model_name=}')
i0 = 0
i_max = 13
chunk_size = 100
precision = 8
layer_names = layers_dict[model_name][1:] # input left aside
class_list = list(class_dict.keys())[:]

crossed_distances = 0
flatten_activations = 0

for key in class_list:
  print(f'class: {key}, code: {class_dict[key]}')
  afolder = f'/scratch/sacevedo/Imagenet2012/train/{class_dict[key]}/a/'
  os.makedirs(figsfolder, exist_ok=True)
  distfolder = distfolder0 + f'{key}/'
  os.makedirs(distfolder, exist_ok=True)
  for layer_id,layer_name in enumerate(layer_names):
    print(f'layer {layer_name}')
    a = load_features_layer(afolder,
                          layer_name,
                          chunk_size,
                          i0=i0,
                          i_max=i_max,
                          flatten=flatten_activations,
                          )
    Ns = a.shape[0]
    if layer_name == 'flatten':
      Nf = 1
      N = a.shape[1]
    else:
      Nf = a.shape[1]
      fsize = a.shape[2]
      N = fsize*fsize
    a = np.round(a,precision)
    a = 2*np.sign(a).astype(int)-1

    if flatten_activations == 0:
      for f_id in range(Nf):
        if layer_name == 'flatten':
          X = a
        else:
          X = np.reshape(a[:,f_id,:,:],(Ns,fsize*fsize))
        print(f'{X.shape=}')
        H = Hamming(coordinates=X,
                    crossed_distances=crossed_distances,)
        H.compute_distances()
        H.D_histogram(compute_flag=1,
                      save=True,
                      Ns=Ns,
                      L=N,
                      t=f_id,
                      resultsfolder=distfolder+f'hist/{layer_name}/',
                      )
    # elif flatten_activations == 0:
    #   pass
      # k_list = np.array(range(a.shape[1])).astype(int)
      # for k in k_list:
      #   X = a[:,k,:,:].reshape(a.shape[0],a.shape[2]*a.shape[3])
      #   print(k,X.shape)
      #   H = Hamming(coordinates=X,
      #               crossed_distances=crossed_distances)
      #   H.compute_distances()
      #   H.D_histogram(compute_flag=1,
      #                 save=True,
      #                 t=k,
      #                 Ns=Ns,
      #                 resultsfolder=dfolder+f'hist/{layer_name}/',
      #                 )



  







