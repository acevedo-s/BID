from aparameters import *

start = time()
assert class_id==-1
key = 'shuffled'

_data = torch.load(shuffled_afolder + f'shuffled_imgs.pt')
print(f'{len(_data)=}')

chunk_size = 100
i0_min = 0
i0_max = len(_data)//chunk_size + 1

normalize = 1
export_cropped_imgs = 0

fmt_function = shuffled_format_resize
actfolder = get_afolder(model_name,key,crop_size)
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

print(f'this took {(time()-start)/60:.1f} mins')










