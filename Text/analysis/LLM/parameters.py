import sys,os
import numpy as np

### HYPERPARAMETERS
Ntokens = 0 # 0 means to take all the tokens in the data sample.
sublength_cutoff = 300 # tokens cutoff for GPU space constraint
# layer_ids = range(25) 
layer_ids = [0,24] # there are 25 for OPT ~300M and also for Pythia

LLM = sys.argv[1]
print(f'{LLM=}')
corpus = sys.argv[2]
print(f'{corpus=}')
randomize = int(sys.argv[3])
print(f'{randomize=}')
batch_randomize = int(sys.argv[4])
print(f'{batch_randomize=}')
Nbits = int(sys.argv[5]) # 0 for real-valued activations # 1 for sign binarization, 2 for alternative binarization
print(f'{Nbits=}')

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
  N_batches = None
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
histfolder = f'results/{corpus}/{LLM}/hist/'
if randomize:
  histfolder += f'randomize/'
if Ntokens != 0:
  histfolder += f'Ntokens{Ntokens}/'
if batch_randomize:
  histfolder += f'Lconcat{Lconcat}/'
if Nbits > 1:
  histfolder += f'Nbits{Nbits}/'
###---
  
### OPTIMIZATION
export_logKLs = 1
optfolder0 = f'results/{corpus}/{LLM}/opt/'
if randomize:
  optfolder0 += f'randomize/'
if Ntokens != 0:
  optfolder0 += f'Ntokens{Ntokens}/'
if batch_randomize:
  optfolder0 += f'Lconcat{Lconcat}/'
if Nbits > 1:
  optfolder0 += f'Nbits{Nbits}/'
###---

# ### GENERATED TEXT
# genfolder0 = f'results/{corpus}/{LLM}/gen/'
# if randomize:
#   genfolder0 += f'randomize/'
# ###---
  
### ACTIVATION HISTS
hist_actfolder = f'results/{corpus}/{LLM}/hist_act/'
if batch_randomize:
  hist_actfolder += f'Lconcat{Lconcat}/'

# ### ACTIVATION CDFs
# cdf_actfolder = f'results/{corpus}/{LLM}/cdf_act/'
# if batch_randomize:
#   cdf_actfolder += f'Lconcat{Lconcat}/'
# cdf_actfolder += f'Nbits{Nbits}/'

### ANGLES
anglesfolder0 = wd + path0 + f'{corpus}/{LLM}/angles/'
if batch_randomize:
  anglesfolder0 += f'Lconcat{Lconcat}/'

signfolder0 = wd + path0 + f'{corpus}/{LLM}/sign/'