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
print(f'{crop_size=}')
resize = int(sys.argv[2])
print(f'{resize=}')

model_id = 0
model_name,W_model = model_list[model_id]
resultsfolder = f'results/{model_name}/'
histfolder = resultsfolder + f'hist/crop_size{crop_size}/'
if resize==0:
  histfolder += f'padding/'
os.makedirs(resultsfolder, exist_ok=True)

print(f'{model_name=}')
i0 = 0
i_max = 13
chunk_size = 100
precision = 8
layer_names = layers_dict[model_name][:] 
class_list = list(class_dict.keys())[:]

crossed_distances = 0
flatten_activations = 1

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
layer_name = layer_names[task_id]

for key_id,key in enumerate(class_list):
  print(f'class: {key}, code: {class_dict[key]}')
  afolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/'
  if resize==0:
    afolder += f'padding/'
  print(f'layer {layer_name}')
  a = load_features_layer(afolder,
                        layer_name,
                        chunk_size,
                        i0=i0,
                        i_max=i_max,
                        flatten=flatten_activations,
                        )
  Ns,N = a.shape
  if key_id==0:
    EDfile = resultsfolder + f'act_shape_{layer_name}.txt'
    # print(f'{N}',
    #       file=open(EDfile, 'a'))
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
  H = Hamming(coordinates=a,
              crossed_distances=crossed_distances)
  H.compute_distances()
  H.D_histogram(compute_flag=1,
                save=True,
                Ns=Ns,
                resultsfolder=histfolder+f'{key}/{layer_name}/',
                )

print(f'this took {(time()-start)/60:.1f} mins')



  







