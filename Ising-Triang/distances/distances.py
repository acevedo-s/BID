import numpy as np
import sys,os
from dadapy import Hamming
from time import time
from T_list import T_list

start = time()
geometry = 'triangular'
eps = 1E-7
metric = 'hamming'
crossed_distances = 0
L = int(sys.argv[1])
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
T_id = task_id
T = T_list[T_id]

N = L**2
datafolder = f'/scratch/sacevedo/Ising-{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
print(f'{T=:.2f}')
X = np.loadtxt(f'{datafile}').astype(int)
Ns = X.shape[0]
print(f'{X.shape=}')
H = Hamming(coordinates=X,
            crossed_distances=crossed_distances)
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              T=T,
              L=L,
              Ns=Ns,
              )
print(f'this took {(time()-start)/60:.2f} minutes')
