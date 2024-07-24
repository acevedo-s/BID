import numpy as np
import sys,os
from dadapy import Hamming
from time import time

start = time()
geometry = 'Ising-Chain'
eps = 1E-7
metric = 'hamming'

L = int(sys.argv[1])
T = float(sys.argv[2])
print(f'{T=:.2f}')
R = int(sys.argv[3])
print(f'{R=}')
Ns0 = 5000            # sampled configurations
Ns = 2500             # only using these to make it faster
Rlim = 50             #500.000 spin chains, Ns0 samples of each

datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'
histfolder = f'results/R{R}/'

if R <= Rlim:
  _R = R
else:
  _R = Rlim

### loading data up to 500.000 spins
for r_id in range(_R):
  datafile = datafolder + f'r_id{r_id}/T{T:.2f}.txt'
  if r_id == 0:
    X = np.loadtxt(datafile).astype(int)
  else:
    X = np.concatenate((X,
                        np.loadtxt(datafile)
                        ),
                      axis=1
                      )

### for the second 500.000 we take the (unused) second half of the samples from the first 500.000
if R <= Rlim:
  X = X[:Ns,:]
else:
  X = np.reshape(X,newshape=(Ns,-1))[:,:R*L]
print(f'{X.shape=}')

H = Hamming(coordinates=X)
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              T=T,
              L=L,
              Ns=Ns,
              resultsfolder=histfolder,
              )
print(f'this took {(time()-start)/60:.2f} minutes')
