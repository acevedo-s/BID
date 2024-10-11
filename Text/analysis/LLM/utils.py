import torch
import os
from time import time
import matplotlib.pyplot as plt
import numpy as np 
from numba import jit 

eps = 1E-7

def ranks_real_to_spin(real_dist_indices,
                     spin_distances,
                     neighbour,
                     resultsfolder,
                     export_hists=0,
                     n_export=10,
                     filename='ranks_RS',
                     ):
  """ 
  Defining ranks such that we have max_rank = Ns-1, as in real-space. 
  neighbour = 1 means first neighbour.
  """
  
  Ns = spin_distances.shape[0]
  R = np.empty(shape=(Ns,)) # ranks in spin space 

  for sample_idx in range(Ns):
    D_values, D_counts = np.unique(spin_distances[sample_idx,:], return_counts=True)
    assert D_values[0] == 0  # trivial zero
    D_counts[0] -= 1
    if D_counts[0] == 0:
      D_values = D_values[1:]
      D_counts = D_counts[1:]
    if export_hists:
      if sample_idx < n_export:
        np.savetxt(fname=f'{resultsfolder}hist{sample_idx}.txt',
        X=np.transpose([D_values,D_counts]))

    neighbour_idx = real_dist_indices[sample_idx,neighbour]
    histogram_idx = np.where(D_values == spin_distances[sample_idx,neighbour_idx])[0][0] 
    if histogram_idx == 0: # first neighbours are rank 0 
      R[sample_idx] = 0
    else:
      R[sample_idx] = np.sum(D_counts[:histogram_idx])
  np.savetxt(fname=f'{resultsfolder}{filename}.txt',X=np.transpose([R]),fmt='%d')
  return


def ranks_spin_to_real(real_dist_indices,
                     spin_distances,
                     neighbour,
                     resultsfolder,
                     filename='ranks_SR'
                     ):
  Ns = spin_distances.shape[0] # number of samples
  Rs = [] # ranks in spin space 
  assert neighbour == 1 # for now the following only works for first neighbour, because of "min_dist"

  for sample_idx in range(Ns):
    min_dist = np.min(np.delete(spin_distances[sample_idx,:],sample_idx)) # minimum hamming distance from sample, excluding the self-distance...
    neighbour_idcs = np.where(spin_distances[sample_idx,:] == min_dist)[0] # list of indeces that share the minimum distance in spin spaces (variable size)
    for neighbour_idx in neighbour_idcs:
      if neighbour_idx == sample_idx: 
        continue # excluding self-distance if its there
      R = np.where(real_dist_indices[sample_idx,:] == neighbour_idx)[0][0] - 1 # typically self distance is index 0, except for some cases (degenerated real-distances?) in which the routines put the degenerated distance at the first possition. for those, rank gives -1.
      assert R != -1; f'BUG: {sample_idx=} has rank -1'
      Rs.append(R) # rank of neighbour_idx in real space. note that minimum rank is 0
  Rs = np.array(Rs)
  np.savetxt(fname=f'{resultsfolder}{filename}.txt', X=np.array(Rs),fmt='%d')
  return

def formatting_activations(a,sub_length,Ns0,layer_normalize):
  """
  Ns0 is the total number of samples loaded: Ns0 = N_batches * batch_size
  """
  # keeping only sub_length tokens
  a = a[:,:sub_length,:]

  # checking for possible repetitions, note that this function reorders data. 
  a = np.unique(a,axis=0)

  B,T,E = a.shape

  if B != Ns0:
    print(f'WARNING: There were repetitions in the real-valued activations. applying "np.unique" reorders data')

  print(f'{a.shape=}')

  # vectorizing activations
  a  = np.reshape(a,(B,sub_length*E))

  # normalizing first layer
  if layer_normalize:
    a = poor_mans_layer_norm(a)

  return a

@jit
def _poor_mans_layer_norm(a,Ns,layer_mean,layer_std):
  for sample_idx in range(Ns):
    a[sample_idx] = (a[sample_idx] - layer_mean[sample_idx]) / layer_std[sample_idx] 
  return a

def poor_mans_layer_norm(a):
  layer_mean = np.mean(a,axis=1)
  layer_std = np.sqrt(np.var(a,axis=1)+ eps)
  print(f'{layer_mean=}')
  print(f'{layer_std=}')
  print(f'applying poor mans layer norm')
  Ns = a.shape[0]
  a = _poor_mans_layer_norm(a,Ns,layer_mean,layer_std)
  return a

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
                     layer_id,
                     LLM,
                     Ntokens,
                     ):
  start = time()
  for batch_id in range(N_batches):
    a_filename = f'{act_outputfolder}b{batch_id}_l{layer_id}.pt'
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
                batch_size,
                act_outputfolder,
                LLM,
                Ntokens,
                remove_activations,
                Nbits,
                ):
  for layer_id in layer_ids:
    print(f'{layer_id=:d}')
    a = load_activations(N_batches,
                        act_outputfolder,
                        layer_id,
                        LLM,
                        Ntokens).numpy()
    B,T,E, = a.shape
    
    if layer_id == 0:
      a = poor_mans_layer_norm(a)

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
                                            layer_id,
                                            t)
      for b in range(B):
        spins_batch = _binarization(a,a_mean,a_std,b,t,spins_batch)
      np.save(sigmas_filename,spins_batch)
    print(f'for each value of t, {spins_batch.shape=}') 
    print(f'binarization took {(time()-start)/60:.1f} m')

    if remove_activations:
      print(f'update code here to remove activations if wanted...')
      # for batch_id in range(N_batches):
      #   a_filename = f'{act_outputfolder}b{batch_id}_l{layer_id}.pt'
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
                layer_id,
                T,
                keep_unique_spins=0,
                ):
  for t in range(T):
    sigmas_filename = get_sigmas_filename(sigmasfolder0,
                                          sublength_cutoff,
                                          layer_id,
                                          t)
    spins_batch = np.load(sigmas_filename)
    # print(f'{spins_batch.shape=}')
    if t == 0:
      sigmas = spins_batch
    else:
      sigmas = np.concatenate((sigmas,spins_batch),axis=1)
  # sigmas = sigmas.astype(np.int8)
  print(f'{sigmas.shape=}')
  if keep_unique_spins:
    sigmas = np.unique(sigmas,axis=0) # note that this re-orders data! carefullll
  return sigmas

def get_sigmas_filename(sigmasfolder0,
                        sublength_cutoff,
                        layer_id,
                        t):
  sigmasfolder = f'{sigmasfolder0}sub_length{sublength_cutoff:d}/'
  os.makedirs(sigmasfolder,exist_ok=True)
  return f'{sigmasfolder}sigmas_l{layer_id}_T{t}.npy'

def get_angles_filename(anglesfolder0,
                        t,
                        tau,
                        layer_id):
  anglesfolder = anglesfolder0 + f't{t}/layer_id{layer_id}/'
  os.makedirs(anglesfolder,exist_ok=True)
  angles_filename = anglesfolder + f'tau{tau}.txt'
  return angles_filename

