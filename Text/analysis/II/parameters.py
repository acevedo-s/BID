import sys,os
import numpy as np

###TODO: agregar layer_id al path de los ranks.


### HYPERPARAMETERS
Ntokens = 0 # 0 means to take all the tokens in the data sample.
sublength_cutoff = 300 # tokens cutoff for GPU space constraint
# layer_ids = range(25) 
layer_ids = [24] # there are 25 for OPT ~300M and also for Pythia

LLM = sys.argv[1]
print(f'{LLM=}')
corpus = sys.argv[2]
print(f'{corpus=}')
randomize = 0 #int(sys.argv[3])
print(f'{randomize=}')
batch_randomize = 0 #int(sys.argv[4])
print(f'{batch_randomize=}')
Nbits = 1 #int(sys.argv[5]) # 0 for real-valued activations # 1 for sign binarization, 2 for alternative binarization
print(f'{Nbits=}')
layer_id = int(sys.argv[3])
print(f'{layer_id=}')
assert layer_id in layer_ids
sub_length = int(sys.argv[4])
print(f'{sub_length=}')

if LLM == 'OPT':
  max_length = 401
elif LLM == 'Pythia':
  max_length = 400

emb_dims = np.array([1024 for _ in range(25)])
if LLM == 'OPT' and layer_ids[-1] == 24 :
  emb_dims[-1] = 512

wd = os.environ['WORK']
path0 = "/sacevedo/Data/Text/"

### TOKENS
tokens_outputfolder0 = wd + path0 + f'{corpus}/{LLM}/input_tokens/'
tokens_outputfolder = f'{tokens_outputfolder0}/max_length{max_length:d}/'
if randomize:
  tokens_outputfolder += 'randomize/'
os.makedirs(tokens_outputfolder,exist_ok = True)
###---

### BATCH RANDOMIZE
if batch_randomize:
  Lconcat = 150
else:
  Lconcat = None

### ACTIVATIONS
compute_activations = 0
remove_activations = 0
print(f'{remove_activations=}')
batch_size = 100
if sublength_cutoff == 300:
  N_batches = 5
elif sublength_cutoff == 10:
  N_batches = 2

act_outputfolder0 = wd + path0 + f'{corpus}/{LLM}/activations/'
act_outputfolder0 = f'{act_outputfolder0}max_length{max_length:d}/'
if randomize:
  act_outputfolder0 += f'randomize/'
if Ntokens != 0:
  act_outputfolder0 += f'Ntokens{Ntokens}/'
if batch_randomize:
  act_outputfolder0 += f'Lconcat{Lconcat}/'
os.makedirs(act_outputfolder0,exist_ok=True)
###---

### SPINS
sigmasfolder0 = wd + path0 + f'{corpus}/{LLM}/sigmas/max_length{max_length:d}/'
if randomize:
  sigmasfolder0 += f'randomize/'
if Ntokens != 0:
  sigmasfolder0 += f'Ntokens{Ntokens}/'
if batch_randomize:
  sigmasfolder0 += f'Lconcat{Lconcat}/'
if Nbits > 1:
  sigmasfolder0 += f'Nbits{Nbits}/'
###---

### DISTANCES 
remove_spins = 0
distfolder = f'results/{corpus}/{LLM}/dists/layer_id{layer_id}/'

if randomize:
  distfolder += f'randomize/'
if Ntokens != 0:
  distfolder += f'Ntokens{Ntokens}/'
if batch_randomize:
  distfolder += f'Lconcat{Lconcat}/'
if Nbits > 1:
  distfolder += f'Nbits{Nbits}/'
###---
  
