from dadapy import Hamming
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *

np.set_printoptions(precision=7)
histfolder = f'../distances/results/hist/'
_optfolder = f'results/opt/'
eps = 1E-7
crossed_distances = 0
Ts = 0.745
precision_T = 3
Ns = 5000

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.3f}')
alphamax = float(sys.argv[3])
print(f'{alphamax=:.5f}')

alphamin = 1E-2 # order of quantile for P(r)
delta = 5E-4
Nsteps = int(1E6)
seed = 1

H = Hamming(crossed_distances=crossed_distances)
H.D_histogram(
              T=T,
              precision_T=precision_T,
              L=L,
              Ns=Ns,
              resultsfolder=histfolder,
              )

### loading previous result for smoothness and regularization
load_T_flag = 0
if load_T_flag:
  T_load = .7
  optfolder0 = _optfolder + f'L{L}/T{T_load:.3f}/'
else:
  optfolder0 = _optfolder + f'L{L}/T{T:.3f}/'

B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        )
if load_T_flag:
  B.load_initial_condition()

# running optimization
B.optfolder = _optfolder + f'L{L}/T{T:.3f}/'
B.computeBID()
