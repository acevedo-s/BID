import numpy as np
import os,sys
import numba

resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-triangular'

@numba.njit
def Energy(X):
  Ns,L,L = X.shape
  E = np.zeros(shape=Ns)
  for i in range(L):
    for j in range(L):
      E = E + X[:,i,j] * X[:,(i+1)%L,j]
      E = E + X[:,i,j] * X[:,i,(j+1)%L]
      E = E + X[:,i,j] * X[:,(i+1)%L,(j-1)%L]
  return E

nseeds = 5000
L_list = [80,90,100]
T_list = np.arange(.1,2+eps,.1)
# T_list = np.array([1])
for L_id,L in enumerate(L_list):
  print(f'{L=}')
  datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'
  E = np.empty(shape=(nseeds,len(T_list)))
  M = np.empty(shape=(nseeds,len(T_list)))
  for T_id,T in enumerate(T_list):
    print(f'{T=:.2f}')
    datafile = datafolder + f'T{T:.2f}.txt'
    X = np.loadtxt(f'{datafile}')[:nseeds]
    Ns,N = X.shape
    X = np.reshape(X,(Ns,L,L))
    if T_id==0:print(f'{X.shape=}')
    E[:,T_id] = Energy(X)

  E_mean = np.mean(E,axis=0)
  Ef = (np.mean(E**2,axis=0)-E_mean**2) 
  np.savetxt(fname=resultsfolder + f'E_L{L}.txt',
             X=np.transpose([T_list,E_mean,Ef]),
             )

  


