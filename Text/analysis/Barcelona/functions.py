import json
import pickle
import sys,os
import torch
import numpy as np
import matplotlib.pyplot as plt



def get_weights_folder(LLM,layer_idx,layer_name):
  weigthsfolder = f'weights/{LLM}/layer{layer_idx}/{layer_name}/'
  os.makedirs(weigthsfolder,exist_ok=True)
  return weigthsfolder

def extract_MLP(output,layer_idx,f,resultsfolder):
  if f == 'fc2':
    ax = 1
  if f == 'fc1':
    ax = 0

  if f == 'f':
    X = np.tensordot(output[layer_idx]['fc1']["weight"],
                     output[layer_idx]['fc2']["weight"],
                     axes=([0],[1])
    )
    eigenvalues, eigenvectors = np.linalg.eig(X)
  else:
    X = np.tensordot(output[layer_idx][f]["weight"],
                     output[layer_idx][f]["weight"],
                     axes=([ax],[ax])
    )
    eigenvalues, eigenvectors = np.linalg.eigh(X)
    np.savetxt(fname=f'{resultsfolder}bias.txt',
               X=output[layer_idx][f]["bias"])

  print(f'{X.shape=}')
  np.savetxt(fname=f'{resultsfolder}eigenvalues.txt',X=eigenvalues)
  np.savetxt(fname=f'{resultsfolder}eigenvectors.txt',X=eigenvectors)

  