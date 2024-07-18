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

class_list = list(class_dict.keys())[:]
i0_min = 0
i0_max = 13
chunk_size =100

crop_size0 = 224
n_blocks = 8
block_size = crop_size0//n_blocks

resultsfolder = f'/scratch/sacevedo/Imagenet2012/shuffled/'
os.makedirs(resultsfolder, exist_ok=True)

for key_id,key in enumerate(class_list):
  print(f'class: {key}, code: {class_dict[key]}')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  print(f'{len(files)=}')
  for i0 in range(i0_min,i0_max):
    if i0*chunk_size<=len(files):
      _data = load_chunk(files,
                    crop_size=crop_size0,
                    chunk_size=chunk_size,
                    i0=i0,
                    normalize=0,
                    )
      if key_id==0:
        X = _data
      else:
        X = torch.cat((X,_data),axis=0) # X.shape = (Nsamples,3,crop_size0,crop_size0)
Nsamples = X.shape[0]
print(f'{X.shape=}')

Y = torch.empty(size=(Nsamples*n_blocks**2,
                      3,
                      block_size,
                      block_size)
                )
idx = 0
for n in range(Nsamples):
  for i in range(n_blocks):
    for j in range(n_blocks):
      Y[idx] = X[n,:,i*block_size:(i+1)*block_size,j*block_size:(j+1)*block_size]
      idx +=1
del X # to free some memory
print(f'{Y.shape=}')
Y = Y[torch.randperm(Nsamples*n_blocks**2),:,:,:] # shuffling patches

X = torch.zeros(size=(Nsamples,3,crop_size0,crop_size0))

idx = 0
for n in range(Nsamples):
  for i in range(n_blocks):
    for j in range(n_blocks):
      X[n,
        :,
        i*block_size:(i+1)*block_size,
        j*block_size:(j+1)*block_size
        ] = Y[idx]
      idx += 1

### visual check:
fig,ax = plt.subplots(1)
ax.imshow(X[0].permute(1,2,0))
fig.savefig(f'results/figs/shuffled{block_size}.pdf')  

### saving shuffled data
print(f'{X.shape=}')
torch.save(X,resultsfolder + f'shuffled_imgs.pt')

