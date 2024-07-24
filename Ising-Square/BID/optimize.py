from dadapy import Hamming
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *

np.set_printoptions(precision=7)
histfolder = f'../distances/results/hist/'
_optfolder = f'results/opt/'
eps = 1E-7
crossed_distances = 0
Ns = 5000

L = int(sys.argv[1])
print(f'{L=}')
T = float(sys.argv[2])
print(f'{T=:.2f}')
alphamax = float(sys.argv[3])
print(f'{alphamax=:.5f}')

alphamin = 0#1E-3
delta = 5E-4
Nsteps = int(1E6)
seed = 1

H = Hamming(crossed_distances=crossed_distances)
H.D_histogram(
              T=T,
              L=L,
              Ns=Ns,
              resultsfolder=histfolder,
              )

optfolder0 = _optfolder + f'L{L}/T{T:.2f}/'

B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        )
B.computeBID()
