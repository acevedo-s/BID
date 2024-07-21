from PIL import Image
from torchvision import transforms
from .resnet import *
import matplotlib.pyplot as plt

import glob
import os,sys
import random
import numpy as np

class_dict = {
  'vizsla' : 'n02100583',
  'koala' : 'n01882714',
  'Shih-Tzu' : 'n02086240',
  'Rhodesian_ridgeback' : 'n02087394',
  'English_setter' : 'n02100735',
  'cabbage_butterfly' : 'n02280649',
  'Yorkshire_terrier'  : 'n02094433',
}


# labels_filename = 'labels.txt'
# if not os.path.isfile(labels_filename):

#   url_labels = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
#   from urllib.request import urlopen
#   from shutil import copyfileobj
#   with urlopen(url_labels) as in_stream, open(labels_filename, 'wb') as out_file:
#       copyfileobj(in_stream, out_file)
#   print(f'Imagenet labels downloaded and sent to {labels_filename}')
  

def fix_files(datafolder,
              resultsfolder,
              filename,
              shuffle=False,
              ):
    """ to be used only once, and keep an ordering of files """
    
    files = glob.glob(datafolder+filename+'/*.JPEG') # paths
    if shuffle:
      random.shuffle(files)

    # os.system(f'rm -f {filename}')

    with open(resultsfolder+filename, 'w') as f:
        for i in range(len(files)):
            f.write(files[i] + '\n')
    return

def load_files(filename):
    with open(filename) as f:
      files = f.read().splitlines()
    return files

def format_torch(filename,
                 crop_size=224, # default in torch vision
                 export_cropped_imgs=False,
                 normalize=1,
                 ):
    """
    Format proposed by PyTorch
    https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
    """
    input_image = Image.open(filename)
    
    ### check original:
    if export_cropped_imgs:
      t0 = transforms.Compose([
          transforms.Resize(224),
          transforms.ToTensor(),
      ])  
      x0 = t0(input_image)
      print(f'{x0.shape=}')
      fig0,ax0 = plt.subplots(1,figsize=(4,4))
      plt.axis('off')
      ax0.imshow(x0.permute(1,2,0))
      fig0.savefig('results/figs/nocrop.pdf')

    if normalize==0:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Resize(224), # we fix the size to this size, for every crop
      ])      
    elif normalize==1:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(224), # we fix the size to this size, for every crop
      ])  
    input_tensor = preprocess(input_image)

    # check crop
    if export_cropped_imgs:
      fig,ax = plt.subplots(1,figsize=(4,4))
      ax.imshow(input_tensor.permute(1,2,0))
      plt.axis('off')
      plt.tight_layout(pad=0)
      fig.savefig(f'results/figs/crop{crop_size}.pdf', 
                  # bbox_inches='tight',
                  )
    return input_tensor

def padding_format_torch(filename,
                        crop_size=224, # default in torch vision
                        export_cropped_imgs=False,
                        normalize=1,
                        ):
    """
    Format proposed by PyTorch
    https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
    """
    input_image = Image.open(filename)
    
    ### check original:
    if export_cropped_imgs:
      t0 = transforms.Compose([
          transforms.Resize(224),
          transforms.ToTensor(),
      ])  
      x0 = t0(input_image)
      print(f'{x0.shape=}')
      fig0,ax0 = plt.subplots(1,figsize=(4,4))
      plt.axis('off')
      ax0.imshow(x0.permute(1,2,0))
      fig0.savefig('results/figs/nocrop.pdf')

    if normalize==0:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Pad(padding=(224-crop_size)//2,
                       fill=0), # we fix the size to this size, for every crop
      ])      
    elif normalize==1:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Pad(padding=(224-crop_size)//2,
                       fill=0), # we fix the size to this size, for every crop
      ])  
    input_tensor = preprocess(input_image)

    # check crop
    if export_cropped_imgs:
      fig,ax = plt.subplots(1,figsize=(4,4))
      ax.imshow(input_tensor.permute(1,2,0))
      plt.axis('off')
      plt.tight_layout(pad=0)
      fig.savefig(f'results/figs/padcrop{crop_size}.pdf', 
                  # bbox_inches='tight',
                  )
    return input_tensor

def find_indices_to_discard(files,chunk_size=None):
    """
    Some images in the dataset are in black and white, and have only one channel,
    this routine finds them.
    """
    if chunk_size==None:
      chunk_size = len(files)

    i = 0
    indices_to_remove = []
    while(i < len(files) and i < chunk_size):
      try:
        format_torch(files[i])
      except:
        print(f'datafile {i} could not be preprocessed')
        indices_to_remove.append(i)
      i += 1

    return indices_to_remove

def remove_BW_files(indices_to_remove,
                    filename):
  files = load_files(filename)

  with open(filename, 'w') as f:
    for line_id, line in enumerate(files):
      if line_id not in indices_to_remove:
        f.write(line + '\n')
  return

def load_chunk(files,
               crop_size,
               chunk_size=None,
               i0=0,
               export_cropped_imgs=False,
               normalize=1,
               resize=None,
               ):
    """

    Inputs:

    files: list of all paths to files
    chunk_size: number of files to process simultaneously
    i0: index to start counting images

    """
    if chunk_size == None:
      chunk_size = len(files)
      
    i = i0*chunk_size
    data = []


    if resize==1:
       fmt_function = format_torch
    elif resize==0:
       fmt_function = padding_format_torch
    else:
       sys.exit('define fmt_function')

    while(i < len(files) and i < (i0+1)*chunk_size):
        try:
            data.append(fmt_function(files[i],
                                     crop_size,
                                     export_cropped_imgs,
                                     normalize
                                     )
                        )
            i+=1
        except:
            print(f'first use \'load_data\' one time to eliminate images with wrong format ')
    
    data = torch.stack(data)
    print(f'data shape: {data.shape}')

    return data

########################## shuffled data:

def shuffled_format_resize(X, #pytorch tensor
                          crop_size=224, # default in torch vision
                          export_cropped_imgs=False,
                          normalize=1,
                          ):
    """
    Format proposed by PyTorch
    https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
    """    
    if normalize==0:
      preprocess = transforms.Compose([
        transforms.CenterCrop(crop_size,
                              ),
        transforms.Resize(224,
                          antialias=False, # to work on tensors
                          ), # we fix the size to this size, for every crop
      ],
)      
    elif normalize==1:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size,
                              # antialias=False, # to work on tensors
                              ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(224), # we fix the size to this size, for every crop
      ])  
    input_tensor = preprocess(X)

    # check crop
    if export_cropped_imgs:
      # plt.tick_params(left = False, right = False , labelleft = False , 
      #           labelbottom = False, bottom = False) 
      # plt.xticks([])
      # plt.yticks([])
      fig,ax = plt.subplots(1,figsize=(4,4))
      plt.axis('off')
      plt.tight_layout(pad=0)
      # ax.set_title(r'$N_c={crop_size**2}$')
      ax.imshow(input_tensor[0].permute(1,2,0))
      fig.savefig(f'results/figs/Xcrop{crop_size}.pdf')
    return input_tensor

def shuffled_format_pad(X, #pytorch tensor
                        crop_size=224, # default in torch vision
                        export_cropped_imgs=False,
                        normalize=1,
                        ):
    """
    Format proposed by PyTorch
    https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py
    """    
    if normalize==0:
      preprocess = transforms.Compose([
        transforms.CenterCrop(crop_size,
                              ),
        transforms.Pad(padding=(224-crop_size)//2,
                       fill=0), # we fix the size to this size, for every crop
      ],
)      
    elif normalize==1:
      preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop_size,
                              # antialias=False, # to work on tensors
                              ),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Pad(padding=(224-crop_size)//2,
                       fill=0), # we fix the size to this size, for every crop
      ])  
    input_tensor = preprocess(X)

    # check crop
    if export_cropped_imgs:
      # plt.tick_params(left = False, right = False , labelleft = False , 
      #           labelbottom = False, bottom = False) 
      # plt.xticks([])
      # plt.yticks([])
      fig,ax = plt.subplots(1,figsize=(4,4))
      plt.axis('off')
      plt.tight_layout(pad=0)
      # ax.set_title(r'$N_c={crop_size**2}$')
      ax.imshow(input_tensor[0].permute(1,2,0))
      fig.savefig(f'results/figs/Xpadcrop{crop_size}.pdf')
    return input_tensor