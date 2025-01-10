import sys,os
sys.path.append('../../')
from paths import *
from dadapy import data
import numpy as np
from time import time 

start = time()

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
M_flag = int(sys.argv[3])
print(f'{M_flag=}')

N = L**2
geometry = 'Ising-square'
datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
X = np.loadtxt(f'{datafile}').astype(int)

resultsfolder = makefolder(base=f'results/gride/',
                           create_folder=True,
                           L=L,
                           T=T,
                           )

if M_flag:
  M0 = np.sum(X,axis=1)
  indices = np.where(M0<0)
  X[indices] = - X[indices]

### GRIDE
_data = data.Data(coordinates=X, maxk=X.shape[0]-1)
range_max = X.shape[0] - 1
ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(range_max=range_max)

### EXPORTING
filename='gride.txt'
np.savetxt(resultsfolder+filename,np.transpose([ids_gride,
                                                ids_err_gride,
                                                rs_gride]))


print(f'this took {(time()-start)/60:.1f} mins')