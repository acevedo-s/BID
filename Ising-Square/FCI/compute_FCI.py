import sys,os
sys.path.append('../../')
from paths import *
import pyFCI
import numpy as np
from time import time 

start = time()

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
M_flag = int(sys.argv[3])
print(f'{M_flag=}')
global_flag = int(sys.argv[4])
print(f'{global_flag=}')


resultsfolder = makefolder(base=f'results/FCI/',
                           create_folder=True,
                           precision=2,
                           global_flag=global_flag,
                           L=L,
                           T=float(T),
                           M_flag=M_flag,
                           )

N = L**2
geometry = 'Ising-square'
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
X = np.loadtxt(f'{datafile}').astype(int)

if M_flag:
  M0 = np.sum(X,axis=1)
  indices = np.where(M0<0)
  X[indices] = - X[indices]

if global_flag == 1:
  X = pyFCI.center_and_normalize(X)
  fci = pyFCI.FCI(X)
  d,x0,err = pyFCI.fit_FCI(fci)
  if np.isclose(T,2.) or np.isclose(T,3.) or np.isclose(T,1.8) or np.isclose(T,2.3):
    np.savetxt(resultsfolder + 'fci.txt',X=fci)
  np.savetxt(resultsfolder + 'FCI_fit.txt',X=[d,x0,err])
else:
  ks = [10]
  for center in range(10):
    r = pyFCI.local_FCI(X,center,ks) # normalizes inside
    d,x0,err = r[0,2:]
    np.savetxt(resultsfolder + f'FCI_fit_center{center}_ks{ks}.txt',X=[d,x0,err])

print(f'{d=}')
print(f'{x0=}')
print(f'{err=}')
print(f'this took {(time()-start)/60:.1f} mins')