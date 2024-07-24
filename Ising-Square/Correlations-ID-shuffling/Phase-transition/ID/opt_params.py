import numpy as np
from dadapy import Hamming
import os,sys
eps = 1E-6


# T_list = np.array([2.,2.1])
# T_list = np.concatenate((T_list,
#                         np.arange(2.2,2.4+eps,.01))
#                         )
# T_list = np.concatenate((T_list,
#                         np.arange(2.5,3+eps,.1))
#                         )
T_list = [2.,2.1,2.5,2.6]#np.arange(2.,3+eps,.1)
T_list = [2.1,2.2,2.3,2.4,2.5]
alphamax_list = np.arange(.1,.3+eps,.1) # orders of quantile for P(r)

if sys.argv[0] == 'opt_params.py':
  print('exporting T_list')
  T_filename = 'T_list.txt'
  os.system(f'rm -f {T_filename}')
  for T in T_list:
    print(f'{T:.2f}',file=open(T_filename,'a'))

  print('exporting alphamax_list')
  alphamax_filename = 'alphamax_list.txt'
  os.system(f'rm -f {alphamax_filename}')
  for alphamax in alphamax_list:
    print(f'{alphamax:.5f}',file=open(alphamax_filename,'a'))