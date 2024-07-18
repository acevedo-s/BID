import sys,os
from transformers import AutoTokenizer
from time import time 
### HYPERPARAMETERS
LLM = sys.argv[1]
print(f'{LLM=}')
corpus = sys.argv[2]
print(f'{corpus=}')
randomize = int(sys.argv[3])
print(f'{randomize=}')

if LLM == 'OPT':
  max_length = 401 # +1 for BOS token...
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
                                device_map="auto",
                                )
elif LLM == 'Pythia':
  max_length = 400 # No BOS token...
  from transformers import (GPTNeoXForCausalLM,
                            GPTNeoXConfig
                            )
  modelname = "EleutherAI/pythia-410m-deduped"
  model = GPTNeoXForCausalLM.from_pretrained(
                            modelname,
                            revision="main",
                            # cache_dir="./pythia-410m-deduped/main",
                            )
  config = GPTNeoXConfig.from_pretrained(modelname,
                                        output_hidden_states=True,
                                        )
  
tokenizer = AutoTokenizer.from_pretrained(modelname,
                                          padding_side='left',
                                          device_map="auto",
                                          )

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
print(vars(model),
      file=open(f'{LLM}_model.txt','w'))
  ###---


wd = os.environ['WORK']
path0 = "/sacevedo/Data/Text/"
### TOKENS
tokens_outputfolder0 = wd + path0 + f'{corpus}/{LLM}/input_tokens/'
tokens_outputfolder = f'{tokens_outputfolder0}/max_length{max_length:d}/'
if randomize:
  tokens_outputfolder += 'randomize/'
os.makedirs(tokens_outputfolder,exist_ok = True)

