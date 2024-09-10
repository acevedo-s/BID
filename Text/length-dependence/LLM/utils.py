import torch
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np 

def batch_shuffle(x, Lconcat, seed=1):
  torch.manual_seed(seed)

  batch_size, L = x.shape
  if Lconcat >= L:
    return x
  else:
    print(f'randomizing after token {Lconcat}!')
    # Split the tensor into two parts at position Lconcat
    first_part = x[:, :Lconcat]
    second_part = x[:, Lconcat:]
    
    # Shuffle the second part along the batch dimension
    indices = torch.randperm(batch_size)
    shuffled_second_part = second_part[indices]
    
    # Recombine the tensor
    shuffled_x = torch.cat((first_part, shuffled_second_part), dim=1)
    
    return shuffled_x
    """
    # Example of usage
    tensor = torch.tensor([
      [1, 2, 3, 4, 5],
      [6, 7, 8, 9, 10],
      [11, 12, 13, 14, 15],
      [16, 17, 18, 19, 20]
    ])
    Lconcat = 2
    seed = 42  # Set a specific seed
    shuffled_tensor = batch_shuffle(tensor, Lconcat, seed)
    print(shuffled_tensor)

    # Re-run to check reproducibility
    shuffled_tensor_again = batch_shuffle(tensor, Lconcat, seed)
    print("Shuffled Tensor Again (should match):")
    print(shuffled_tensor_again)
    """

def load_activations(N_batches,
                     act_outputfolder,
                     l_id,
                     LLM,
                     Ntokens,
                     ):
  start = time()
  for batch_id in range(N_batches):
    a_filename = f'{act_outputfolder}b{batch_id}_l{l_id}.pt'
    if batch_id == 0:
      a = torch.load(a_filename,map_location=torch.device('cpu'))
    else:
      a = torch.cat((a,
                    torch.load(a_filename,map_location=torch.device('cpu')))
                    )
  if LLM == 'OPT':
    a = a[:,1:,:] # Every sentence starts with the BOS=2 token in OPT
  a = a[:,-Ntokens:,:] # in case we want to retain only Ntokens
  print(f'importing took {(time()-start)/60:.1f} m')
  print(f'{a.shape=}')
  return a

# def activations_CDF(a):
  
#   a = torch.sort(torch.flatten(a))
#   Nact = len(a)
  
#   return a,torch.range(start=1,end=Nact)/Nact

def binarization(
                sigmasfolder0,
                sublength_cutoff,
                layer_ids,
                N_batches,
                act_outputfolder,
                LLM,
                Ntokens,
                remove_activations,
                Nbits,
                ):
  for l_id in layer_ids:
    print(f'{l_id=:d}')
    a = load_activations(N_batches,
                        act_outputfolder,
                        l_id,
                        LLM,
                        Ntokens).numpy()
    B,T,E, = a.shape
    
    start = time()
    spins_batch = None
    if Nbits==2:
      _binarization = _2_bit_binarization
      a_mean = np.mean(a) ; print(f'{a_mean=}')
      a_std = np.std(a) ; print(f'{a_std=}')
    else:
      _binarization = _sign_binarization
      a_mean = None
      a_std = None
    
    for t in range(T):
      print(f'{t=}')
      sigmas_filename = get_sigmas_filename(sigmasfolder0,
                                            sublength_cutoff,
                                            l_id,
                                            t)
      for b in range(B):
        spins_batch = _binarization(a,a_mean,a_std,b,t,spins_batch)
      np.save(sigmas_filename,spins_batch)
    print(f'for each value of t, {spins_batch.shape=}') 
    print(f'binarization took {(time()-start)/60:.1f} m')

    if remove_activations:
      print(f'update code here to remove activations if wanted...')
      # for batch_id in range(N_batches):
      #   a_filename = f'{act_outputfolder}b{batch_id}_l{l_id}.pt'
      #   os.system(f'rm -f {a_filename}')

def _2_bit_binarization(a,a_mean,a_std,b,t,spins_batch):
  first = (a[b,t] < a_mean - a_std)
  second = (a[b,t] < a_mean)
  third = (a[b,t] > a_mean + a_std)
  a[b,t][np.asarray(first).nonzero()] = 0
  a[b,t][np.asarray(second & ~ first).nonzero()] = 1
  a[b,t][np.asarray(~ third & ~ second).nonzero()] = 2
  a[b,t][np.asarray(third).nonzero()] = 3
  del first,second,third
  bits = np.unpackbits(a[b,t].astype(np.uint8)).reshape(-1,8)[:,-2:].flatten() # length: 2*E
  bits = (2*bits-1).astype(np.int8)
  if b == 0:
    spins_batch = bits
  else:
    spins_batch = np.vstack((spins_batch,bits))
  return spins_batch

""" the following code snippet can be runned to check the 2-bit encoding
import torch
import numpy as np
a = np.array([[0.5,1.2,2,3.5],
              [7,2.5,-1,-4]])
a_mean = np.mean(a)
a_std = np.std(a)
print(f'{a=}')
print(f'{a_mean=}')
print(f'{a_std=}')
B = len(a)
for i in range(B):
  first = (a[i] < a_mean - a_std)
  second = (a[i] < a_mean)v
  third = (a[i] > a_mean + a_std)
  a[i][np.asarray(first).nonzero()] = 0               #00
  a[i][np.asarray(second & ~ first).nonzero()] = 1    #01
  a[i][np.asarray(~ third & ~ second).nonzero()] = 2  #10
  a[i][np.asarray(third).nonzero()] = 3               #11
  print(a[i])
  bits = np.unpackbits(a[i].astype(np.uint8)).reshape(-1,8)[:,-2:].flatten()
  if i ==0:
    sigmas = bits
  else:
    sigmas = np.vstack((sigmas,bits))
print(sigmas)
"""

def _sign_binarization(a,a_mean,a_std,b,t,spins_batch):
  spins = np.sign(a[b,t]).astype(np.int8)
  if b == 0:
    spins_batch = spins
  else:
    spins_batch = np.vstack((spins_batch,spins))
  
  # """ZEROS ARE SET TO -1 (<ALMOST> NEVER HAPPENS)"""
  spins_batch[np.asarray(spins_batch == 0).nonzero()] = -1
  return spins_batch

def load_sigmas(sigmasfolder0,
                sublength_cutoff,
                l_id,
                T,
                ):
  for t in range(T):
    sigmas_filename = get_sigmas_filename(sigmasfolder0,
                                          sublength_cutoff,
                                          l_id,
                                          t)
    spins_batch = np.load(sigmas_filename)
    # print(f'{spins_batch.shape=}')
    if t == 0:
      sigmas = spins_batch
    else:
      sigmas = np.concatenate((sigmas,spins_batch),axis=1)
  # sigmas = sigmas.astype(np.int8)
  print(f'{sigmas.shape=}')
  sigmas = np.unique(sigmas,axis=0)
  return sigmas

def get_sigmas_filename(sigmasfolder0,
                        sublength_cutoff,
                        l_id,
                        t):
  sigmasfolder = f'{sigmasfolder0}sub_length{sublength_cutoff:d}/'
  os.makedirs(sigmasfolder,exist_ok=True)
  return f'{sigmasfolder}sigmas_l{l_id}_T{t}.npy'

def get_angles_filename(anglesfolder0,
                        t,
                        tau,
                        layer_id):
  anglesfolder = anglesfolder0 + f't{t}/layer_id{layer_id}/'
  os.makedirs(anglesfolder,exist_ok=True)
  angles_filename = anglesfolder + f'tau{tau}.txt'
  return angles_filename