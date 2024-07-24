from dadapy import Hamming
import os,sys
import numpy as np
from time import time

np.set_printoptions(precision=7)
histfolder = f'../distances/results/hist/'
_optfolder = f'results/testGD/opt/'
eps = 1E-7
crossed_distances = 0
Ns = 5000

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
alphamax = float(sys.argv[3])
print(f'{alphamax=:.5f}')

alphamin = 0 # order of quantile for P(r)
Nsteps = int(5E5)
seed = 1
for L in [100,90,80,70,60,50,40,30]:
  for T in np.arange(.1,4+eps,.1):
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
                  T=T,
                  L=L,
                  Ns=Ns,
                  resultsfolder=histfolder,
                  )
    optfolder0 = _optfolder + f'L{L}/T{T:.2f}/'
    if False:
      start = time()
      from dadapy._utils.stochastic_minimization_hamming import *
      delta = 1E-4
      B = BID(H,
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder0,
            )
      B.computeBID()
      print(f'{B.Op.sigma/L**2=}')
      del B
      print(f'this took {(time()-start):.1f} secs')

    if True:
      Nsteps = int(2E5)
      start = time()
      from dadapy._utils.gradient_descent_hamming import *
      eta0 = jnp.double(1E-4)
      eta1 = jnp.double(1E-4)
      B = BID(
              N=L**2,
              H=H,
              alphamin=alphamin,
              alphamax=alphamax,
              seed=seed,
              eta0=eta0,
              eta1=eta1,
              Nsteps=Nsteps,
              optfolder0=optfolder0,
              )
      B.computeBID()
      print(f'this took {(time()-start):.1f} secs')
      
