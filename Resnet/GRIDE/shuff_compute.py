import sys,os
sys.path.append('../')
from R import load_features_layer
from R.models import *
from R.data import *
from R.paths import *
from dadapy import data
import numpy as np
from time import time 
start = time()

resize = int(os.getenv('resize'))
print(f'{resize=}')

crop_size = int(sys.argv[1])
print(f'{crop_size=}')
class_id = int(sys.argv[2])
print(f'{class_id=}')
layer_id = int(sys.argv[3])
print(f'{layer_id=}')
dbg=int(sys.argv[4])
print(f'{dbg=}')

model_id = 0
model_name,W_model = model_list[model_id]
print(f'{model_name=}')
layer_names = layers_dict[model_name][:] 
class_list = list(class_dict.keys())[:]

layer_name = layer_names[layer_id]
print(f'layer {layer_name}')

key = 'shuffled'
print(f'{key=}')

### loading activations:
i0 = 0
if dbg:
  i_max = 1
else:
  i_max = 79 # can be seen in the datafolder...

chunk_size = 100
flatten_activations = 1

afolder = get_afolder(model_name,key,crop_size)
a = load_features_layer(afolder,
                      layer_name,
                      chunk_size,
                      i0=i0,
                      i_max=i_max,
                      flatten=flatten_activations,
                      )
Ns,N = a.shape
EDfile = get_EDfilename('.',model_name,layer_name,key)
np.savetxt(fname=EDfile,X=a.shape,fmt='%d')

### BINARIZATION
precision = 8
a = np.round(a,precision)
a = 2 * np.sign(a).astype(int) - 1

### SEPARATING DATA IN BLOCKS TO HAVE THE SAME NUMBER OF DATA SAMPLES AS THE FIRST EXPERIMENT
n_blocks = len(class_list) # there were 7 classes, so lets divide the dataset in 7 again
block_size = Ns // n_blocks
print(f'{block_size=}')
print(f'{Ns=}')
a = a[class_id*block_size:(class_id+1)*block_size]
print(f'{a.shape=}')

### GRIDE
_data = data.Data(coordinates=a, maxk=a.shape[0]-1)
range_max = a.shape[0] - 1
ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(range_max=range_max)

### EXPORTING
resultsfolder = makefolder(base=f'results/gride/shuffle/',
                           create_folder=True,
                           crop_size=crop_size,
                           key=class_id,
                           layer_id=layer_id,
                           )
filename='gride.txt'
np.savetxt(resultsfolder+filename,np.transpose([ids_gride,ids_err_gride,rs_gride]))

print(f'this took {(time()-start)/60:.1f} mins')



  







