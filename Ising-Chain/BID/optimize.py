from dadapy import Hamming
import os,sys
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *

np.set_printoptions(precision=7)
eps = 1E-7
Ns = 2500


L = int(sys.argv[1])
T = float(sys.argv[2])
R = int(sys.argv[3])
print(f'{R=}')

optfolder0 = f'results/L{L}/T{T:.2f}/Ns{Ns}/'
optfolder0 += f'R{R}/'

alphamin = 0 #1E-3
alphamax = float(sys.argv[4])
print(f'{alphamax=:.5f}')

delta = 5E-4
Nsteps = int(1E6)
seed = 111

histfolder = f'../distances/results/R{R}/'
H = Hamming()
H.D_histogram(
              L=L,
              T=T,
              Ns=Ns,
              resultsfolder=histfolder,
              )
B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        export_logKLs=1,
        L=R*L,
        )
B.computeBID()
