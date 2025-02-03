import os,sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('../../')
from paths import *

# this environmental variable must be set <before> the BID imports, to work with JAX double-precision
os.environ["JAX_ENABLE_X64"] = "True"
from dadapy.hamming import BID, Hamming


L = 100  # system width or height (you can see size-differences putting here L=30)
N = L**2 # total number of spins in a two-dimensional square lattice of length L 
Ns = int(sys.argv[1])
T = float(sys.argv[2])
alphamax = float(sys.argv[3])
print(f'{T=:.2f}')
print(f'{Ns=}')

# PARAMETER DEFINITIONS FOR OPTIMIZATION
seed = 1
alphamin = 0      # order of min_quantile, to remove poorly sampled parts of the histogram if necessary (see Supp. Inf. of paper)
delta = 1e-3        # stochastic optimization step size
Nsteps = int(1e6)   # number of optimization steps
export_results = 1  # flag to export d0,d1,logKL,Pemp,Pmodel after optimization (default=1)
export_logKLs = 1   # flag to export the logKLs during optimization (default=0)
optfolder0 = f"results/opt/L{L}/T{T:.2f}/Ns{Ns}/"  # folder where optimization results are saved
histfolder = f'results/hist/L{L}/T{T:.2f}/Ns{Ns}/'

H = Hamming()
H.D_histogram(compute_flag=0,            # if 0 the histograms are loaded instead of computed
            save=False,                 # we compute the histograms once and save time in the future
            resultsfolder=histfolder,  # folder where the histograms are saved
            )
B = BID(
H=H,
alphamin=alphamin,
alphamax=alphamax,
seed=seed,
delta=delta,
Nsteps=Nsteps,
export_results=export_results,
export_logKLs=export_logKLs,
optfolder0=optfolder0,
L=L**2,
)

B.computeBID()  # results are defined as attributes of B. They are also exported if export_results=1 (default)