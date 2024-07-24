import numpy as np
from dadapy import Hamming
import os,sys
from time import time
eps = 1E-6

histfolder = f'results/hist/'
shuff_histfolder = f'results/shuff/hist/'
Z2flag = 1
Mplusflag = 0

if Z2flag:
  histfolder += 'Z2/'
  shuff_histfolder += 'Z2/'
elif Mplusflag:
  histfolder += 'Mplus/'
  shuff_histfolder += 'Mplus/'

L = 4096
seed = 3
datafolder0 = f'/scratch/sacevedo/Ising-square/canonical/'
crossed_distances = 0


T_list = np.array([2.,2.1])
T_list = np.concatenate((T_list,
                        np.arange(2.2,2.4+eps,.02))
                        )
T_list = np.concatenate((T_list,
                        np.arange(2.5,3.0+eps,.1))
                        )

# T_list = np.arange(2.1,2.5+eps,.1)
# T_list = [2.22]
print(f'{len(T_list)=}')  
if sys.argv[0] == 'params.py':
  print('exporting T_list')
  T_filename = 'T_list.txt'
  os.system(f'rm -f {T_filename}')
  for T in T_list:
    print(f'{T:.2f}',file=open(T_filename,'a'))