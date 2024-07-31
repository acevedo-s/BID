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

# resize = 1 #int(sys.argv[2])
# print(f'{resize=}')
resize = int(os.getenv('resize'))
print(f'{resize=}')

crop_size = int(sys.argv[1])
print(f'{crop_size=}')
class_id = int(sys.argv[2])
print(f'{class_id=}')
layer_id = int(sys.argv[3])
print(f'{layer_id=}')

model_id = 0
model_name,W_model = model_list[model_id]
print(f'{model_name=}')
layer_names = layers_dict[model_name][:] 
class_list = list(class_dict.keys())[:]

# task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
layer_name = layer_names[layer_id]
print(f'layer {layer_name}')
key = class_list[class_id]
print(f'class: {key}, code: {class_dict[key]}')


### loading activations:
i0 = 0
i_max = 13
chunk_size = 100
precision = 8
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
# zeros = np.where(np.isclose(a,0))
# print(len(zeros[0]))
a = np.round(a,precision)
a = 2*np.sign(a).astype(int)-1
# print(f'{a.shape=}')
# zeros = np.where(a==0)
# print(len(zeros[0]))
# print(a)
# print()

H = Hamming(coordinates=a)
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              Ns=Ns,
              resultsfolder=get_histfolder('.',model_name,crop_size,key,layer_name),
              )

print(f'this took {(time()-start)/60:.1f} mins')



  







