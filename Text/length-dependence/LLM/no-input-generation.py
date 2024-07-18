from paths import *
from shared_parameters import *
from transformers import AutoTokenizer
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')
from time import time
import numpy as np

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

os.makedirs(genfolder0,exist_ok=True)
### PARAMETERS 
sample_id = int(sys.argv[4])
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
length = 10*task_id  
if LLM == 'OPT':
  length += 1 # Every sentence starts with a BOS = 2 token in OPT

### MODEL 
if LLM=='OPT':
  from transformers.models.opt import (
                                      OPTForCausalLM,
                                      OPTConfig,
                                      )
  modelname = "facebook/opt-350m"
  config = OPTConfig.from_pretrained(modelname,
                                    output_hidden_states=True,
                                    )
  model = OPTForCausalLM.from_pretrained(modelname,
                                  config=config,
                                  ).to(device)
  print(vars(model),
        file=open(f'{LLM}model.txt','w'))
elif LLM=='Pythia':
  from transformers import (GPTNeoXForCausalLM,
                            GPTNeoXConfig
                            )
  modelname = "EleutherAI/pythia-410m-deduped"
  config = GPTNeoXConfig.from_pretrained(modelname,
                                        output_hidden_states=True,
                                        )
  model = GPTNeoXForCausalLM.from_pretrained(
                                        modelname,
                                        revision="main",
                                        # cache_dir="./pythia-410m-deduped/main",
                                        ).to(device)

### TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(modelname,
                                        device_map="auto",
                                        padding_side='left',
                                        )

### Pythia doesn't have a PAD token...
if LLM == 'Pythia':
  tokenizer.add_special_tokens({'pad_token': '[PAD]'})
max_input_length = config.max_position_embeddings

### TEXT GENERATION
genfile = genfolder0 + f'no-input-gen-length{length}-sample{sample_id}' +'.txt'
start = time()
with torch.no_grad():
  g_output = model.generate(
                          max_length=length,
                          do_sample=True,
                          top_k=10,
                          return_dict_in_generate=True,
                          output_hidden_states=False,
                          )
g_ids = g_output['sequences']
print(f'{g_ids.shape=}')
g = tokenizer.batch_decode(g_ids,
                          skip_special_tokens=True,)
print(f'{g[0]}',file=open(genfile, 'a'))

print(f'generation took {(time()-start)/60:.1f} min')