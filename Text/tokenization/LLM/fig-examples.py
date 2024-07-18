from paths import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')
from transformers import AutoTokenizer
import numpy as np
from transformers.models.opt import (
                                     OPTModel, # no-head
                                     OPTConfig,
                                     )



### MODEL 
modelname = "facebook/opt-350m"
config = OPTConfig.from_pretrained(modelname,
                                  output_hidden_states=True,
                                  )
model = OPTModel.from_pretrained(modelname,
                                config=config,
                                device_map="auto",
                                )
tokenizer = AutoTokenizer.from_pretrained(modelname,
                                        padding_side='left',
                                        device_map="auto",
                                        )

### IMPORT TOKENS
x0 = torch.from_numpy(
                  np.loadtxt(tokens_outputfolder + 'token_ids.txt').astype(int)
                  ).to(device)
print(f'{x0.shape=}')

if randomize==0:
  print_indeces = [414]
  sublengths = [i * 10 + 1 for i in range(1,40)]
else:
  print_indeces = [0,1,2]
  sublengths = [71, 191, 291]
for l in sublengths:
  x_decoded = tokenizer.batch_decode(x0[:,:l],
                                    skip_special_tokens=True)
  for i in print_indeces:
    print(f'{x_decoded[i]}',file=open(f'output_r{randomize}_{i}_{l}.txt','w'))