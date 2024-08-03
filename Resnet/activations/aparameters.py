import os,sys
sys.path.append('../')
from R import *
from time import time


model_id = 0
model_name,W_model = model_list[model_id]
filesfolder = f'../results/files/'

resize = 1                      # first we do a crop, then we resize to fixed original size.

crop_size = int(sys.argv[1])    # the crop will be of size crop_size x crop_size
print(f'{crop_size=}')
class_id = int(sys.argv[2])     # -1 for shuffled data


R = Resnet(model_name,W_model)
R.get_nodes()
print(R.nodes)

layer_names = layers_dict['resnet18']
