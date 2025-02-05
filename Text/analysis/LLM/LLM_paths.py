import sys,os

wd = os.environ['WORK']
path0 = "/sacevedo/Data/Text/"

def make_optfolder0(corpus,LLM,randomize,Ntokens,Lconcat,batch_randomize,Nbits):
  optfolder0 = f'results/{corpus}/{LLM}/opt/'
  if randomize:
    optfolder0 += f'randomize/'
  if Ntokens != 0:
    optfolder0 += f'Ntokens{Ntokens}/'
  if batch_randomize:
    optfolder0 += f'Lconcat{Lconcat}/'
  if Nbits > 1:
    optfolder0 += f'Nbits{Nbits}/'
  return optfolder0

def get_act_outputfolder0(max_length,corpus,LLM,randomize,Lconcat,batch_randomize):

  act_outputfolder0 = wd + path0 + f'{corpus}/{LLM}/activations/'
  act_outputfolder0 = f'{act_outputfolder0}max_length{max_length:d}/'
  if randomize:
    act_outputfolder0 += f'randomize/'
  if batch_randomize:
    act_outputfolder0 += f'Lconcat{Lconcat}/'
  # os.makedirs(act_outputfolder0,exist_ok=True)
  return act_outputfolder0