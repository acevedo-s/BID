import numpy as np
import sys,os
from dadapy import Hamming
from time import time

start = time()
geometry = 'Ising-square'
eps = 1E-7
metric = 'hamming'
crossed_distances = 0
L = int(sys.argv[1])
T = float(sys.argv[2])
half = int(sys.argv[3])

N = L**2
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
print(f'{T=:.2f}')
X = np.loadtxt(f'{datafile}').astype(int)
print(f'{X.shape=}')
if half==1:
  X = X[:,:N//2]
elif half==2:
  X = X[:,N//2:]
Ns = X.shape[0]
print(f'{X.shape=}')
H = Hamming(coordinates=X,
            crossed_distances=crossed_distances)
H.compute_distances()
histfolder=f'results/hist/half{half}/'
H.D_histogram(compute_flag=1,
              save=True,
              T=T,
              L=L,
              Ns=Ns,
              resultsfolder=histfolder,
              )
print(f'this took {(time()-start)/60:.2f} minutes')
