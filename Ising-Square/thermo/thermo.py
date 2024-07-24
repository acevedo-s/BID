import numpy as np
import matplotlib.pyplot as plt
import os

resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-square'
def energy(X):
  Ns,L = X.shape
  E = np.zeros(shape=Ns)
  for i in range(L):
    E -= X[:,i] * X[:,(i+1)%L]
    E -= X[:,i] * X[:,(i+np.sqrt(L).astype(int))%L]
  return E

seed = 1
L_list = [4096]#[1024]
T_list = np.arange(.05,4+eps,.05)#np.arange(.1,4+eps,.1)
# T_list = np.concatenate((np.arange(2.2,2.5+eps,.01),
#                          np.arange(2.6,4+eps,.1)))
for L_id,L in enumerate(L_list):
  datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}_seed_{seed}/'
  hyperp = np.loadtxt(datafolder + 'hyperp.txt')
  Ns = hyperp[6].astype(int)
  E = np.zeros(shape=(Ns,len(T_list)))
  for T_id,T in enumerate(T_list):
    datafile = datafolder + f'T{T:.2f}.npy'
    X = np.load(f'{datafile}')[:Ns,:]
    if T_id==0:print(f'{X.shape=}')
    E[:,T_id] = energy(X)

  E_mean = np.mean(E,axis=0)
  Cv = (np.mean(E**2,axis=0)-E_mean**2) / T_list**2

S = np.array([np.trapz(Cv[:idx]/T_list[:idx],T_list[:idx]) / np.log(2) 
     for idx in range(0,len(T_list))]) / L 
E_mean /= L
Cv /= L

if True:
  np.savetxt(fname=f'results/thermo_N{L}_seed_{seed}.txt',
             X=np.transpose([T_list,S,E_mean,Cv]),fmt='%1.3f')
  fig = plt.figure()
  ax = plt.subplot(111)
  plt.plot(T_list,E_mean+2,'o',label='E/L + E_0')
  plt.plot(T_list,Cv,'o',label='Cv/L')
  plt.plot(T_list,S,'o',label='S/L')
  Tc_id = np.where(np.isclose(Cv,np.max(Cv)))[0][0]
  plt.vlines(T_list[Tc_id],
             plt.ylim()[0],
             plt.ylim()[1],
             color='gray')
  plt.xlabel('T')
  plt.legend()
  plt.savefig(fname=f'{resultsfolder}thermo.png')
  plt.close

  


