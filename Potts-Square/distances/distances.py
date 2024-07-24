import numpy as np
import sys,os
from dadapy import Hamming
from time import time
from T_list import *

precision_T = 3
np.set_printoptions(precision=precision_T)

def q2b(X,Ns,N,q):
  Y = np.empty(shape=(Ns,N,int(np.log2(q))),
                dtype=int)
  for i in range(Ns):
    for j in range(N):
      Y[i,j,:] = np.array([*bin(X[i][j])[2:].zfill(3)]).astype(int)
  return Y 

geometry = 'Potts-square'
eps = 1E-7
metric = 'hamming'
crossed_distances = 0

L = int(sys.argv[1])
print(f'{L=}')
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
T_id = task_id

### reduced T_list:
# T_list = [.1,.746,.5,.6,.7,1,2,3,4]
# T_list = [.5,.6,1,2]
# T = T_list[T_id]

N = L**2
T = T_list[T_id]

start = time()
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.3f}.txt'
print(f'{T=:.3f}')
X = np.loadtxt(f'{datafile}').astype(int)
print(f'{X[:10,:10]}')
Ns,_ = X.shape
assert _ == N

# collecting ground states:
G0filename = f'results/G0_L{L}.txt'
if T == .1:
  G0 = (np.round(np.mean(X,axis=1))).astype(int)
  np.savetxt(fname=G0filename,X=G0,fmt='%d')
else:
  G0 = np.loadtxt(G0filename).astype(int)
# we make all simulations converge to zero with a symmetry transformation:
X = np.mod(X-np.reshape(G0,(Ns,1)),8)

print(f'{X.shape=}')
print(f'{X[:10,:10]}')
Y = q2b(X,Ns,N,q=8)
Y = 2*Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2])-1
print(f'{Y.shape=}')
print(f'{Y[:10,:10*3]}')
H = Hamming(coordinates=Y,
            crossed_distances=crossed_distances)
H.compute_distances(check_format=False)
H.D_histogram(compute_flag=1,
              save=True,
              T=T,
              L=L,
              Ns=Ns,
              precision_T=precision_T,
              )
print(f'this took {(time()-start)/60:.3f} minutes')
