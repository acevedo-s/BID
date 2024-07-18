from R import *
import sys,os
figsfolder = f'results/figs/'
filesfolder = f'results/files/'
os.makedirs(figsfolder, exist_ok=True)
model_id = 0
model_name,W_model = model_list[model_id]
# print(model,W_model)

shuffled_datafolder = f'/scratch/sacevedo/Imagenet2012/shuffled/'
_data = torch.load(shuffled_datafolder + f'shuffled_imgs.pt')
print(f'{len(_data)=}')

crop_size = int(sys.argv[1]) ### 224 is the maximum
# print(f'{crop_size=}')
resize = int(sys.argv[2])

i0 = 0
X = _data[i0:i0+1]
if resize:
  print(f'{X.shape=}')
  X = shuffled_format_resize(X,
                  crop_size,
                  export_cropped_imgs=1,
                  normalize=0,
                  )
else:
  print(f'{X.shape=}')
  X = shuffled_format_pad(X,
                  crop_size,
                  export_cropped_imgs=1,
                  normalize=0,
                  )











