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

shuffled_datafolder = f'/scratch/sacevedo/Imagenet2012/shuffled/'
_data = torch.load(shuffled_datafolder + f'shuffled_imgs.pt')
print(f'{len(_data)=}')
R = Resnet(model_name,W_model)
R.get_nodes()
# print(R.nodes)
layer_names = layers_dict['resnet18'][:]

chunk_size = 100
i0_min = 0
i0_max = len(_data//chunk_size)
crop_size = int(sys.argv[1]) ### 224 is the maximum
print(f'{crop_size=}')
resize = int(sys.argv[2])
print(f'{resize=}')
actfolder = f'/scratch/sacevedo/Imagenet2012/act/shuffled/crop_size{crop_size}/'

normalize = 1
export_cropped_imgs = 0

if resize:
  fmt_function = shuffled_format_resize
else:
  fmt_function = shuffled_format_pad
  actfolder += f'padding/'
os.makedirs(actfolder, exist_ok=True)

for i0 in range(i0_min,i0_max):
  if i0*chunk_size<=len(_data):
    X = _data[i0*chunk_size:(i0+1)*chunk_size]
    X = fmt_function(X,
                    crop_size,
                    export_cropped_imgs=export_cropped_imgs,
                    normalize=normalize,
      )
    for layer_id,layer_name in enumerate(layer_names):
      print(f'{i0=}, processing {layer_name}')
      R.extract_features_from_layer(X,
                                  layer_name,
                                  actfolder,
                                  i0,
                                  chunk_size,
                                  )

print(f'finished!, this took {(time()-start)/60:.1f} mins')










