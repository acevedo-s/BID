import sys,os
sys.path.append('../')
from R import load_features_layer
from R.models import *
from R.data import *
from R.paths import * 
from dadapy import Hamming
import numpy as np
from time import time 
start = time()

crop_size = int(sys.argv[1])
print(f'{crop_size=}')
class_id = int(sys.argv[2])
layer_id = int(sys.argv[3])


model_id = 0
model_name,W_model = model_list[model_id]

class_list = list(class_dict.keys())[:]
key = 'shuffled'
print(f'{key=}')

n_blocks = 7 # there were 7 classes, so lets divide the dataset in 7 again
i0 = 0
i_max = 79 # can be seen in the datafolder...

chunk_size = 100
precision = 8
layer_names = layers_dict[model_name][:] 
layer_name = layer_names[layer_id]
afolder = get_afolder(model_name,key,crop_size)
print(f'layer {layer_name}')

flatten_activations = 1

a = load_features_layer(afolder,
                      layer_name,
                      chunk_size,
                      i0=i0,
                      i_max=i_max,
                      flatten=flatten_activations,
                      )
Ns,N = a.shape
EDfile = get_EDfilename('.',model_name,layer_name,key)
np.savetxt(fname=EDfile,
X=a.shape,fmt='%d')

a = np.round(a,precision)
a = 2*np.sign(a).astype(int)-1

block_size = Ns // n_blocks
print(f'{block_size=}')
print(f'{Ns=}')
for block_id in range(n_blocks):
  X = a[block_id*block_size:(block_id+1)*block_size]
  print(f'{X.shape=}')
  H = Hamming(coordinates=X)
  H.compute_distances()
  H.D_histogram(compute_flag=1,
                save=True,
                resultsfolder=get_histfolder('.',model_name,crop_size,key,layer_name),
                r_id=block_id,
                )

print(f'this took {(time()-start)/60:.1f} mins')



  







