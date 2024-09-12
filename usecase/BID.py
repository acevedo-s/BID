# git clone https://github.com/acevedo-s/DADApy-develop.git
# move to repo folder 
# python setup.py install

import numpy as np
import sys,os
from dadapy import Hamming
from dadapy._utils.stochastic_minimization_hamming import *
import matplotlib.pyplot as plt

# DATA
L = 100    # number of bits 
Ns = 5000  # number of samples
X = 2*np.random.randint(low=0,high=2,size=(Ns,L))-1 # spins must be normalized to +-1 

histfolder = f'results/hist/' #folder where distance histograms are saved
try:
  # LOADING DISTANCES (IF ALREADY COMPUTED)
  H = Hamming()
  H.D_histogram(
              compute_flag=0,
              save=False,
              L=L,
              Ns=Ns,
              resultsfolder=histfolder,
              )
except:
  # COMPUTING DISTANCES 
  H = Hamming(coordinates=X)
  H.compute_distances()
  H.D_histogram(compute_flag=1,
                save=True,
                L=L,
                Ns=Ns,
                resultsfolder=histfolder,
                )

### OPTIMIZATION

# PARAMETER DEFINITIONS
eps = 1E-5                   # good-old small epsilon
alphamin = 0 #+ eps          # order of  min_quantile, to remove poorly sampled parts of the histogram
alphamax = 1 #- eps          # order of max_quantile, to define r* (named rmax here)
delta = 5E-4                 # stochastic optimization step 
Nsteps = int(1E6)            # number of optimization steps
seed = 1                     # 
optfolder0 = f'results/opt/' # folder where optimization results are saved
export_logKLs = 1            # flag to export the logKLs during optimization

B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        export_logKLs=export_logKLs,
        optfolder0=optfolder0,
        L=L
        )

# OPTIMIZING OR LOADING OPTIMIZATION
loaded_flag = 0
while loaded_flag==0:
  try:
    # LOADING RESULTS WHEN COMPUTED
    rmax,d0,d1,logKL = B.load_results()
    remp,Pemp,Pmodel = B.load_fit()
    logKLs_opt = B.load_logKLs_opt()
    loaded_flag=1
    print(f'{d0=}')
    print(f'{d1=}')
    print(f'{logKL=}')
  except:
    B.computeBID()
    # rmax,d0,d1,logKL = B.load_results()

### PLOTTING

# MODEL VALIDATION
figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
figm,axm = plt.subplots()

axm.plot(remp/L,
        Pemp,
        'x',
        zorder=0,
        )
axm.plot(remp/L,
        Pmodel,
        zorder=1,
        color='black',
        )
figm.savefig(figsfolder + f'model_validation.pdf',bbox_inches='tight')

# OPTIMIZATION logKLs
figKL,axKL = plt.subplots()
axKL.plot(logKLs_opt,
          color='black',
          )
figKL.savefig(figsfolder + f'logKLs_optimization.pdf',bbox_inches='tight')