import os 

def get_afolder(model_name,key,crop_size):
  actfolder0 = '/scratch/sacevedo/Imagenet2012/act/'
  return actfolder0 + f'{model_name}/{key}/crop_size{crop_size}/'

def get_histfolder(distance_folder,model_name,crop_size,key,layer_name):
  histfolder = f'{distance_folder}/results/{model_name}/hist/crop_size{crop_size}/{key}/{layer_name}/'
  return histfolder

def get_EDfilename(distance_folder,model_name,layer_name,key):
  EDfolder = f'{distance_folder}/results/{model_name}/{key}/'
  os.makedirs(EDfolder, exist_ok=True)
  EDfilename = EDfolder + f'act_shape_{layer_name}.txt'
  return EDfilename

def get_optfolder(optimization_folder,model_name,crop_size,key,layer_name):
  optfolder = f'{optimization_folder}/results/opt/{model_name}/crop_size{crop_size}/class{key}/layer_name{layer_name}/'
  return optfolder

shuffled_afolder = f'/scratch/sacevedo/Imagenet2012/shuffled/'