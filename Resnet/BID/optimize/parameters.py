import sys,os
from R.models import *
from R.data import *
from R.relative_depth import *
from R.paths import * 

crop_size = int(sys.argv[1])
print(f'{crop_size=}')
layer_id = int(sys.argv[2])
print(f'{layer_id=}')
class_id = int(sys.argv[3])
class_list = list(class_dict.keys())
key = class_list[class_id]
print(f'{key}({class_dict[key]}):')

# ### just to get number of samples Ns...
# filesfolder = f'../../results/files/'
# filename = filesfolder + class_dict[key]
# files = load_files(filename=filename)
# Ns = len(files)

model_id = 0
model_name,W_model = model_list[model_id]
layer_names = layers_dict[model_name]
# layer_id = 3 
# 3=peak 
# 7=previous to flatten
layer_name = layer_names[layer_id]
print(f'{layer_name=}')
relative_depth = relative_depth_dict[model_name][layer_id]
print(f'{relative_depth=}')

# histfolder = f'../../distances/results/{model_name}/hist/crop_size{crop_size}/'
distance_folder = '../../distances'
histfolder = get_histfolder(distance_folder,model_name,crop_size,key,layer_name)
optfolder0 = get_optfolder('.',model_name,crop_size,key,layer_name)
# helpfolder0 = f'results/opt/{model_name}/crop_size112/'
EDfile = get_EDfilename(distance_folder,model_name,layer_name,key)