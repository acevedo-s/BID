import torch 
def get_lengths(attention_mask):
  """
  attention_mask.shape = (Ns,max_length)
  Returns a np.array of length Ns with the number of non-pad 
  tokens in each sentece.
  """
  Non_pad_ids =  torch.nonzero(attention_mask,
                               as_tuple=True)
  _,numtokens = torch.unique(Non_pad_ids[0],
                           return_counts=True)
  # print(numtokens)
  # print(numtokens.shape)
  return numtokens