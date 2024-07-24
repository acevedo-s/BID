import numpy as np
import sys,os

Z2flag = 0
Mplusflag = 1

eps = 1E-7
histfolder0 = f'../distances/results/hist/'
histfolder = f'../distances/results/shuff/hist/'

if Z2flag:
  histfolder0 += 'Z2/'
  histfolder += 'Z2/'
elif Mplusflag:
  histfolder0 += 'Mplus/'
  histfolder += 'Mplus/'

rfolder0 = f'results/rs/'
KLsfolder = f'results/KLs/'

crossed_distances = 0
Nr = 100

T_list = np.arange(2.0,2.1+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.2,2.4+eps,.02))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.5,2.5+eps,.1)))
# T_list = [2.]
# T_list = np.arange(2.1,2.5+eps,.1)
N = 4096
# print(T_list)
Ns = 10000

