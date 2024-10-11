from parameters import *
from utils import * 
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')
from time import time,sleep
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

sub_length = sublength_cutoff
if LLM == 'OPT':
  sub_length += 1# Every sentence starts with a BOS = 2 token after OPT tokenizer, not the case for Pythia's...


### MODEL 
if LLM=='OPT':
  from transformers.models.opt import (
                                      OPTModel, # no-head
                                      OPTConfig,
                                      )
  modelname = "facebook/opt-350m"
  config = OPTConfig.from_pretrained(modelname,
                                    output_hidden_states=True,
                                    )
  model = OPTModel.from_pretrained(modelname,
                                  config=config,
                                  ).to(device)
elif LLM=='Pythia':
  from transformers import (GPTNeoXForCausalLM,
                            GPTNeoXConfig
                            )
  modelname = "EleutherAI/pythia-410m-deduped"
  model = GPTNeoXForCausalLM.from_pretrained(
                            modelname,
                            revision="main",
                            # cache_dir="./pythia-410m-deduped/main",
                            ).to(device)
  config = GPTNeoXConfig.from_pretrained(modelname,
                                        output_hidden_states=True,
                                        )
print(vars(model),
      file=open(f'{LLM}model.txt','w'))
###---

### IMPORT TOKENS
x0 = torch.from_numpy(
                  np.loadtxt(tokens_outputfolder + 'token_ids.txt').astype(int)
                  ).to(device)
print(f'{x0.shape=}')
Ns0, _ = x0.shape
# print(f'{x0.shape=}')
# print(x0[0])
# x_decoded = tokenizer.batch_decode(x0,
#                                   skip_special_tokens=True,
#                                   )
# print(x_decoded[0])


### MAX-TOKEN-LENGTH THRESHOLDING
x = x0[:,:sub_length]
print(f'sub-length x shape:{x.shape}')

# print(f'{x=}')
if batch_randomize:
  x = batch_shuffle(x,Lconcat)
# print(f'{x=}')

# sys.exit()

### FORWARD PASS
Ns = x.shape[0]
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
os.makedirs(act_outputfolder,exist_ok=True)
if N_batches == None:
  N_batches = Ns // batch_size
print(f'{Ns=:d} ; {batch_size=:d} ; {N_batches=:d}')
if compute_activations:
  start = time()
  for batch_id in range(N_batches):
    print(f'{batch_id=:d}')
    # y = {}
    # y['input_ids'] = x['input_ids'][batch_size*batch_id:batch_size*(batch_id+1)]
    # y['attention_mask'] = x['attention_mask'][batch_size*batch_id:batch_size*(batch_id+1)]
    y = x[batch_size*batch_id:batch_size*(batch_id+1),:]
    with torch.no_grad():
      output = model(
                    input_ids=y,
                    output_hidden_states=True,
                    return_dict=True,
                    )
    # print(f'{len(output["hidden_states"])=}') # 25
    for layer_id in layer_ids:
      act_filename = f'{act_outputfolder}b{batch_id:d}_l{layer_id:d}.pt'
      torch.save(output['hidden_states'][layer_id],act_filename)

      
  print(f'forward pass took {time()-start:.1f} seconds')
  del output,model

###SPINS:
if binarize_activations:
  binarization(
            sigmasfolder0,
            sublength_cutoff,
            layer_ids,
            N_batches,
            batch_size,
            act_outputfolder,
            LLM,
            Ntokens,
            remove_activations,
            Nbits,
            )
sleep(1)

