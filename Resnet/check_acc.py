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
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)

figsfolder = f'results/figs/'
filesfolder = f'results/files/'
os.makedirs(figsfolder, exist_ok=True)
model_id = 0
model_name,W_model = model_list[model_id]
# print(model,W_model)

### categories from ImageNet
with open("labels.txt", "r") as f:
  categories = [s.strip() for s in f.readlines()]
categories = np.array(categories)


R = Resnet(model_name,W_model)
# R.get_nodes()
# print(R.nodes)
layer_names = layers_dict['resnet18']
class_list = list(class_dict.keys())[5:6]

i0_min = 0
i0_max = 1
chunk_size = 100

for key in class_list:
  resultsfolder = f'results/acc/'
  os.makedirs(resultsfolder, exist_ok=True)
  print(f'class: {key}, code: {class_dict[key]}')
  filename = filesfolder + class_dict[key]
  files = load_files(filename=filename)
  print(f'{len(files)=}')
  for i0 in range(i0_min,i0_max):
    if i0*chunk_size<=len(files):
      data = load_chunk(files,
                        chunk_size=chunk_size,
                        i0=i0)
    ### model prediction on classification
    with torch.no_grad():
      output = R.model(data)
    probabilities = torch.nn.functional.softmax(output,dim=1)#[0], dim=0)
    print(probabilities.shape)
    predictions = torch.argmax(probabilities,dim=1).numpy()
    predicted_class,counts = np.unique(predictions,return_counts=True)
    print(categories[predicted_class])
    print(counts)






# ### First image in "files.txt"
# fig,ax = plt.subplots(1)
# im = Image.open(files[0])
# ax.imshow(im)
# fig.savefig(figsfolder + '0.png')

# ### top five guesses of the network
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#   print(categories[top5_catid[i]], top5_prob[i].item())



  







