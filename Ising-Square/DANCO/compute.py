import sys,os
sys.path.append('../../')
from paths import *
import numpy as np
from time import time 
from skdim.id import DANCo

start = time()

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
M_flag = int(sys.argv[3])
print(f'{M_flag=}')

Ns = int(os.getenv('Ns'))
N = L**2
geometry = 'Ising-square'
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
X = np.loadtxt(f'{datafile}').astype(int)[:Ns]

if M_flag:
  M0 = np.sum(X,axis=1)
  indices = np.where(M0<0)
  X[indices] = - X[indices]

danco = DANCo()
danco.fit(X)
d = danco.dimension_
print(f'{d=}')


resultsfolder = makefolder(base=f'results/ID/',
                           create_folder=True,
                           L=L,
                           T=float(T),
                           M_flag=M_flag,
                           Ns=Ns,
                           )
np.savetxt(resultsfolder + 'd.txt',X=[d])

print(f'this took {(time()-start)/60:.1f} mins')
