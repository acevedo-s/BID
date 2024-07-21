import sys,os
sys.path.append('../')
from R import load_features_layer
from R.models import *
from R.data import *
from dadapy import Hamming
import numpy as np
from time import time 
start = time()

crop_size = int(sys.argv[1])
n_blocks = 7 # there were 7 classes, so lets divide the dataset in 7 again

model_id = 0
model_name,W_model = model_list[model_id]
resultsfolder = f'results/{model_name}/'
histfolder = resultsfolder + f'hist/crop_size{crop_size}/shuffled/'
os.makedirs(resultsfolder, exist_ok=True)
print(f'{model_name=}')
i0 = 0
i_max = 79 # can be seen in the datafolder...

chunk_size = 100
precision = 8
layer_names = layers_dict[model_name][:] 
crossed_distances = 0
flatten_activations = 1

try:
  task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
  task_id = int(sys.argv[2])
layer_name = layer_names[task_id]
afolder = f'/scratch/sacevedo/Imagenet2012/act/shuffled/crop_size{crop_size}/'
print(f'layer {layer_name}')

a = load_features_layer(afolder,
                      layer_name,
                      chunk_size,
                      i0=i0,
                      i_max=i_max,
                      flatten=flatten_activations,
                      )
Ns = a.shape[0]
# zeros = np.where(np.isclose(a,0))
# print(len(zeros[0]))
a = np.round(a,precision)
a = 2*np.sign(a).astype(int)-1
# print(f'{a.shape=}')
# zeros = np.where(a==0)
# print(len(zeros[0]))
# print(a)
# print()
block_size = Ns // n_blocks
print(f'{block_size=}')
print(f'{Ns=}')
for block_id in range(n_blocks):
  X = a[block_id*block_size:(block_id+1)*block_size]
  print(f'{X.shape=}')
  H = Hamming(coordinates=X,
              crossed_distances=crossed_distances)
  H.compute_distances()
  H.D_histogram(compute_flag=1,
                save=True,
                resultsfolder=histfolder+f'{layer_name}/',
                r_id=block_id,
                )

print(f'this took {(time()-start)/60:.1f} mins')



  







