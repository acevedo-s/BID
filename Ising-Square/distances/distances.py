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
try:
  task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
  T_id = task_id
except:
  task_id = None
  T_id = int(sys.argv[2])
  

T_list = np.arange(1,2.2+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.21,2.39+eps,.01))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.4,4+eps,.1)
                         )
                         )
T = T_list[T_id]

### to know how many temperatures there are
if task_id == None: sys.exit(f'{len(T_list)=}')

# T_list = [2,2.3,3,4,np.inf]
N = L**2
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
print(f'{T=:.2f}')
if T < np.inf:
  X = np.loadtxt(f'{datafile}').astype(int)
else:
  Ns_rand = 5000
  X = 2*np.random.randint(low=0,high=2,size=(Ns_rand,N))-1
Ns = X.shape[0]
# Y = []
# for x_id, x in enumerate(X):
#   y = np.reshape(x,(Ns,L,L))[:Ns,:k,:k]
#   Y.append(np.reshape(y,(Ns,k**2)))
# X = Y 
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
