import matplotlib.pyplot as plt
from R import *
import sys,os
from time import time
start = time()
figsfolder = f'results/figs/'
filesfolder = f'results/files/'
os.makedirs(figsfolder, exist_ok=True)
model_id = 0
model_name,W_model = model_list[model_id]
# print(model,W_model)


R = Resnet(model_name,W_model)
R.get_nodes()
print(R.nodes)
layer_names = layers_dict['resnet18'][:] # the input layer is not binarizable...
# layer_names = ['relu','maxpool']
class_list = list(class_dict.keys())[:]

i0_min = 0
i0_max = 20
chunk_size = 100
# crop_size = 224
crop_size = int(sys.argv[1])
print(f'{crop_size=}')
resize = int(sys.argv[2])
print(f'{resize=}')

for key in class_list:
  actfolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/'
  if resize==0:
    actfolder += f'padding/'
  os.makedirs(actfolder, exist_ok=True)
  print(f'class: {key}, code: {class_dict[key]}')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  print(f'{len(files)=}')
  for i0 in range(i0_min,i0_max):
    if i0*chunk_size<=len(files):
      _data = load_chunk(files,
                        crop_size=crop_size,
                        chunk_size=chunk_size,
                        i0=i0,
                        resize=resize,
                        )
      for layer_id,layer_name in enumerate(layer_names):
        print(f'{i0=}, processing {layer_name}')
        R.extract_features_from_layer(_data,
                                    layer_name,
                                    actfolder,
                                    i0,
                                    chunk_size,
                                    )

print(f'finished!, this took {(time()-start)/60:.1f} mins')


  







