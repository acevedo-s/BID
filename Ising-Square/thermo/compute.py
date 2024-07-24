import numpy as np
import os,sys
import numba

Tc = 2.269
E0 = - 2
resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-square'

@numba.njit
def Energy(X):
  Ns,L,L = X.shape
  E = np.zeros(shape=Ns)
  for i in range(L):
    for j in range(L):
      E = E - X[:,i,j] * X[:,(i+1)%L,j]
      E = E - X[:,i,j] * X[:,i,(j+1)%L]
  return E

def Mag(X):
  M = np.abs(np.sum(X,axis=(1,2)))
  return M

nseeds = 2000
L_list = [90]
T_list = np.arange(1,2.1+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.2,2.4+eps,.01))
                         )
T_list = np.concatenate((T_list,
                        # np.array([2.5])
                         np.arange(2.5,4+eps,.1)
                        )
                        )
# T_list = np.array([1])
for L_id,L in enumerate(L_list):
  print(f'{L=}')
  datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'
  # hyperp = np.loadtxt(datafolder + f'hyperp_L{L}.txt')
  E = np.empty(shape=(nseeds,len(T_list)))
  M = np.empty(shape=(nseeds,len(T_list)))
  for T_id,T in enumerate(T_list):
    datafile = datafolder + f'T{T:.2f}.txt'
    X = np.loadtxt(f'{datafile}')[:nseeds]
    Ns,N = X.shape
    X = np.reshape(X,(Ns,L,L))
    if T_id==0:print(f'{X.shape=}')
    E[:,T_id] = Energy(X)
    M[:,T_id] = Mag(X)

  E_mean = np.mean(E,axis=0)
  Ef = (np.mean(E**2,axis=0)-E_mean**2) 
  M_mean = np.mean(M,axis=0)
  Mf = (np.mean(M**2,axis=0)-M_mean**2)
  # print(E_mean/N,np.std(E,axis=0)/N)
  # print(M_mean/N,np.std(M,axis=0)/N)
  # print(M/N)
  np.savetxt(fname=resultsfolder + f'M_L{L}.txt',
             X=np.transpose([T_list,M_mean,Mf]),
             )
  np.savetxt(fname=resultsfolder + f'E_L{L}.txt',
             X=np.transpose([T_list,E_mean,Ef]),
             )

  


