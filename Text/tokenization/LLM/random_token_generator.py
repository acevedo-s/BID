from paths import *
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

Ns_rand = 7500
X = np.zeros(shape=(Ns_rand,max_length),dtype=int)
X[:,0] = config.bos_token_id
X[:,1:] = np.random.randint(low=3,
                            high=config.vocab_size,
                            size=(Ns_rand,max_length-1)
                            )
output_filename = tokens_outputfolder + f'token_ids.txt'
np.savetxt(output_filename,X,fmt='%d')

if True:
  m = 3
  n = 10
  print(X[:m,:n])
  x_decoded = tokenizer.batch_decode(X[:m,:n],
                                    skip_special_tokens=False)
  print(x_decoded)

