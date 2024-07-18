from paths import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device=}')

from datasets import load_dataset
from time import time
import numpy as np
from utils import get_lengths

# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.simplefilter("ignore", FutureWarning)


### DATASET + CHARACTER THRESHOLDING
start = time()
if corpus == 'Wikitext':
  wt = load_dataset('wikitext','wikitext-103-raw-v1',split='train')
  wt = wt.filter(lambda x: len(x["text"]) > 2*max_length) # removing white spacings and crap
  if False:
    wt = wt.select(range(10000))
  texts = wt['text']
# elif corpus=='Tinystories':
#   path0 = '/scratch/sacevedo/Tinystories'
#   wt = load_dataset(path0,
#                     data_files='TinyStoriesV2-GPT4-train.txt',
#                     split='train[:15%]')
#   wt = wt.filter(lambda x: x["text"] != ' ' and x["text"] != '')
#   texts = wt['text'][0]
#   for idx,element in enumerate(wt['text'][1:]):
#     # if element == ' ': print('spaceeee')
#     # if element == '': print('empty?')
#     # texts.append(element)
#     texts+=(element)
#   texts = texts.split('<|endoftext|>')
elif corpus == 'OWebtext':
  wt = load_dataset('stas/openwebtext-10k','plain_text',split='train')
  texts = wt['text']
print(f'loading took {time()-start:.1f} sec')

if True:
  for i in range(1):
    # print(texts[i])
    print(f'{i=}',texts[i])
print(f'{len(texts)=:d}')
###---


###TOKENIZATION
start = time()
max_input_length = config.max_position_embeddings
# print(f'{config.max_position_embeddings=}')
# max_input_length = max_length
x = tokenizer(texts,
              return_tensors="pt",
              padding=True,
              truncation=True, 
              max_length=max_input_length,
              ).to(device)
del texts, wt
print(f'tokenization took {time()-start:.1f} seconds')
print(f'{x["input_ids"].shape=}')
print(f'{x["attention_mask"].shape=}')
###---
print('--------------------------------------------------')
# print(f'{x["input_ids"]=}')
# print(f'{x["attention_mask"]=}')

### MIN-TOKEN-LENGTH THRESHOLDING
print(f'MIN-TOKEN-LENGTH THRESHOLDING:')
Ns0,max_length_in_batch = x["input_ids"].shape
numtokens = get_lengths(x['attention_mask'])
selected_ids = torch.where(numtokens>max_length,1,0)
x['input_ids'] = x['input_ids'][selected_ids.nonzero(as_tuple=True)]
x['attention_mask'] = x['attention_mask'][selected_ids.nonzero(as_tuple=True)]
print(f'{x["input_ids"].shape=}')
print(f'{x["attention_mask"].shape=}')
# x_decoded = tokenizer.batch_decode(x['input_ids'],
#                                   skip_special_tokens=True)
# print(x_decoded[0])
print(f'-------------------------------------------------')

### MAX-TOKEN-LENGTH THRESHOLDING
print(f'MAX-TOKEN-LENGTH THRESHOLDING:')
numtokens = get_lengths(x['attention_mask'])
Ns,max_length_in_batch = x["input_ids"].shape
numpads = max_length_in_batch - numtokens
z = torch.empty(size=(Ns,max_length),dtype=torch.int)
for i in range(Ns):
  z[i] = x["input_ids"][i,numpads[i]:numpads[i]+max_length]
print(f'{z.shape=}')
print(f'{z=}')
print(f'-------------------------------------------------')

### REMOVING REPETITIONS
print(f'REMOVING REPETITIONS:')
z = torch.unique(z,dim=0) # this reorders the sentences
print(f'{z.shape=}')
Ns,_ = z.shape
assert _ == max_length
print(f'-------------------------------------------------')

### RANDOMIZING
if randomize:
  print('RANDOMIZING:')
  if LLM == 'OPT':
    z = z[:,1:]
    z = z.reshape(Ns*(max_length-1))[torch.randperm(Ns*(max_length-1))]
    z = z.reshape(Ns,(max_length-1))
    z = torch.cat((2*torch.ones(size=(Ns,1),dtype=int),z),
                          dim=-1)
  elif LLM == 'Pythia':
    z = z.reshape(Ns*(max_length))[torch.randperm(Ns*(max_length))]
    z = z.reshape(Ns,(max_length))
  print(f'{z=}')
np.savetxt(tokens_outputfolder + f'token_ids.txt',z,fmt='%d')