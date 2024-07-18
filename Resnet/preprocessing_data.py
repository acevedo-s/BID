import matplotlib.pyplot as plt
from R import *
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


datafolder = f'/scratch/sacevedo/Imagenet2012/train/'
resultsfolder = f'results/files/'
figsfolder = f'results/figs/'
os.makedirs(figsfolder, exist_ok=True)
os.makedirs(resultsfolder, exist_ok=True)

model_id = 0
model_name,W_model = model_list[model_id]

class_list = list(class_dict.keys())[4:6+1]
# if not os.path.isfile(filename):
for key in class_list:
  # to be used only once.
  filename = class_dict[key]
  print(f'class: {key}, code: {filename}')
  chunk_size = None
  fix_files(datafolder,
            resultsfolder,
            filename) 
  files = load_files(resultsfolder+filename)
  print(f'{len(files)=}')
  indices_to_remove = find_indices_to_discard(files,
                                              chunk_size=chunk_size)
  print(indices_to_remove)
  remove_BW_files(indices_to_remove,resultsfolder+filename)
  files = load_files(resultsfolder+filename)


# ### Check first image in "files.txt"
# chunk_size = 5
# files = load_files(resultsfolder+filename)
# data = load_chunk(files,
#                   chunk_size=chunk_size,
#                   i0=0)
# fig,ax = plt.subplots(1)
# im = Image.open(files[0])
# ax.imshow(im)
# fig.savefig(figsfolder + '0.png')

# R = Resnet(model_name,W_model)


# ### model prediction on classification
# with torch.no_grad():
#   output = R.model(data)
# probabilities = torch.nn.functional.softmax(output[0], dim=0)

# ### categories from ImageNet
# with open("labels.txt", "r") as f:
#   categories = [s.strip() for s in f.readlines()]
# # print(categories)

# ### top five guesses of the network
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#   print(categories[top5_catid[i]], top5_prob[i].item())