from dadapy import Hamming
import os,sys
import numpy as np
from routines import * 

np.set_printoptions(precision=7)
histfolder = f'../Ising-Square/distances/results/hist/'
resultsfolder = f'results/BID/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-7
Ns = 5000

L = int(sys.argv[1])
print(f'{L=}')
alphamin = float(sys.argv[2])
print(f'{alphamin=:.2f}')
alphamax = float(sys.argv[3])
print(f'{alphamax=:.2f}')


T_list = np.arange(2.3,3.01,.1)
T_list = [2.0,2.1,2.2]
for T_id,T in enumerate(T_list):

  H = Hamming()
  H.D_histogram(
                T=T,
                L=L,
                Ns=Ns,
                resultsfolder=histfolder,
                )

  H.set_r_quantile(alphamin)
  r0 = H.r
  r0_idx = H.r_idx

  H.set_r_quantile(alphamax)
  r1 = H.r
  r1_idx = H.r_idx

  probs = H.D_probs[r0_idx:r1_idx+1].astype(float)
  rs = H.D_values[r0_idx:r1_idx+1].astype(float)

  ds = np.zeros(shape=len(probs)-1)
  for idx in range(len(probs)-1):
    ds[idx] = MLE_shells(idx,rs,probs)
  np.savetxt(resultsfolder + f'T{T:.2f}.txt',np.transpose([rs[:-1],ds,probs[:-1]]))
