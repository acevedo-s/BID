import numpy as np
import os,sys
import numba

resultsfolder = 'results/correlations/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-triangular'

@numba.njit
def twoCorr(X,i0,j0):
  Nr,L,L = X.shape
  C = np.zeros(shape=(L//2,L//2))
  for r in range(Nr):
    for j in range(L//2):
      for i in range(L//2):
        C[i,j] = C[i,j] + X[r,i0,j0] * X[r,(i0+i)%L,(j0+j)%L]
  return C / Nr


L = int(sys.argv[1])
i0 = int(sys.argv[2])
j0 = int(sys.argv[3])

nseeds = 5000
T_list = np.arange(.1,1+eps,.1)
T_ist = np.array([.1])

print(f'{L=}')
print(f'{i0=}')
print(f'{j0=}')

datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'
for T_id,T in enumerate(T_list):
  print(f'{T=:.2f}')
  datafile = datafolder + f'T{T:.2f}.txt'
  X = np.loadtxt(f'{datafile}')[:nseeds]
  Ns,N = X.shape
  X = np.reshape(X,(Ns,L,L))
  if T_id==0:print(f'{X.shape=}')
  if True:
    C = twoCorr(X,i0,j0)
    np.savetxt(resultsfolder + f'Corr_i0_{i0}_j0_{j0}_T{T:.2f}_L{L}.txt',
          C)
  # Ci0 = twoCorr_i0(X)
  # np.savetxt(resultsfolder + f'Corr_i0_T{T:.2f}_L{L}.txt',
  #         Ci0)





