import numpy as np
from parameters import * 
from dadapy import Hamming
from dadapy._utils.stochastic_minimization_hamming import *

alphamin = float(sys.argv[4])
alphamax = float(sys.argv[5])

delta = 5E-4
Nsteps = int(1E6)
seed = 1

### to get the number of spins, for the initial condition of the optimization
Ns,N = np.genfromtxt(EDfile,
                    dtype='str',
                    unpack=True).astype(int)
print(f'{N=},{Ns=}')

H = Hamming()
H.D_histogram(Ns=Ns,
              resultsfolder=histfolder)
H.compute_moments()
# print(f'{H.D_values=}')
# print(f'{H.D_mu_emp=}')
# print(f'{H.D_var_emp=}')

B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        export_logKLs=0,
        L=N
        )
B.computeBID()