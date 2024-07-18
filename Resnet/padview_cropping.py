import matplotlib.pyplot as plt
from R import *
import sys,os
from time import time
start = time()
#for fancy plotting
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)

figsfolder = f'results/figs/'
filesfolder = f'results/files/'
os.makedirs(figsfolder, exist_ok=True)
model_id = 0
model_name,W_model = model_list[model_id]
# print(model,W_model)

c = 5
class_list = list(class_dict.keys())[c:c+1]
i0_min = 1
i0_max = 2
chunk_size = 1
crop_size = int(sys.argv[1])

for key in class_list:
  resultsfolder = f'/scratch/sacevedo/Imagenet2012/act/{key}/crop_size{crop_size}/'
  os.makedirs(resultsfolder, exist_ok=True)
  print(f'class: {key}, code: {class_dict[key]}')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  print(f'{len(files)=}')
  for i0 in range(i0_min,i0_max):
    if i0*chunk_size<=len(files):
      _ = load_chunk(files,
                    crop_size=crop_size,
                    chunk_size=chunk_size,
                    i0=i0,
                    export_cropped_imgs=1,
                    normalize=0, # because we want to see the image using Imshow
                    resize=0, # because we are padding
                    )
      

