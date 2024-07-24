import numpy as np
import matplotlib.pyplot as plt
import os,sys
import numba

resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5
geometry = 'Ising-Chain'


@numba.njit
def Energy(X):
  Ns,L = X.shape
  E = np.zeros(shape=Ns)
  for i in range(L):
    E = E - X[:,i] * X[:,(i+1)%L]
  return E

def Mag(X):
  M = np.abs(np.sum(X,axis=(1)))
  return M

L = int(sys.argv[1])
T = float(sys.argv[2])
R = int(sys.argv[3])

T_list = [T]
datafolder = f'/scratch/sacevedo/{geometry}/canonical/L{L}/'


# r_id = 0
# datafile = datafolder + f'r_id{r_id}/T{T:.2f}.txt'
# X = np.loadtxt(datafile).astype(int)
# r_id = 1
# datafile = datafolder + f'r_id{r_id}/T{T:.2f}.txt'
# Y = np.loadtxt(datafile).astype(int)
# print( np.isclose(X,Y) )


for r_id in range(R):
  datafile = datafolder + f'r_id{r_id}/T{T:.2f}.txt'
  if r_id == 0:
    X = np.loadtxt(datafile).astype(int)
  else:
    X = np.concatenate((X,
                        np.loadtxt(datafile)
                        ),
                      axis=1
                      )
Ns,N = X.shape


E = np.zeros(shape=(Ns,len(T_list)))
M = np.zeros(shape=E.shape)
for T_id,T in enumerate(T_list):
  if T_id==0:print(f'{X.shape=}')
  E[:,T_id] = Energy(X)
  M[:,T_id] = Mag(X)
E_mean = np.mean(E,axis=0)
# M_mean = np.mean(M,axis=0)
# E_mean /= N
print(f'{E_mean/N=}')
# print(f'{M_mean/N=}')


  # Cv = (np.mean(E**2,axis=0)-E_mean**2) / T_list**2

# S = np.array([np.trapz(Cv[:idx]/T_list[:idx],T_list[:idx]) / np.log(2) 
#      for idx in range(0,len(T_list))]) / L 
# Cv /= L

if False:
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

  


