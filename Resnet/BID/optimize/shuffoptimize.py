from dadapy import Hamming
from parameters import * 
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *
from R.relative_depth import * 

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


""" For homogeneity I divided the "patches" dataset in 7 equivalent subgroups with
 ~ 1200 samples each, since we were using 7 classes with ~ 1200 data points each"""
try:
  r_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
except:
  r_id = int(sys.argv[6])

H = Hamming()
H.D_histogram(
  r_id=r_id,
  resultsfolder=histfolder,
)

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