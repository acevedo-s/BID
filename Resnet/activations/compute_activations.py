from aparameters import *
from time import time

start = time()

R = Resnet(model_name,W_model)
R.get_nodes()
print(R.nodes)
layer_names = layers_dict['resnet18'] # the input layer is not binarizable...
class_list = list(class_dict.keys())
key = class_list[class_id]

chunk_size = 100 # we process up to chunk_size images in simultaneous
i0_min = 0       # min index to start count images in a chunk
i0_max = 20      # max index to start counting images in a chunk (if its too big it doesn't matter)

actfolder = get_afolder(model_name,key,crop_size)
os.makedirs(actfolder, exist_ok=True)
print(f'class: {key}, code: {class_dict[key]}')
filename = filesfolder + class_dict[key]
files = load_files(filename=filename)
Ns = len(files)
print(f'{Ns=}')

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

print(f'this took {(time()-start)/60:.1f} mins')


  







