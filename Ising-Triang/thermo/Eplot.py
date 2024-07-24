import numpy as np
import matplotlib.pyplot as plt
import os
import numba

#for fancy plotting
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)


resultsfolder = 'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5

L_list = [30,60,100]#,40,60,100]
figE,axE = plt.subplots(1)
figfE,axfE = plt.subplots(1)

start = 0
for L_id,L in enumerate(L_list):
  T_list,E_mean,fE = np.loadtxt(resultsfolder + f'E_L{L}.txt',unpack=True)[:,start:]
  fE /= T_list**2 * L**2
  E_mean /= L**2
  axE.plot(T_list,E_mean,
           'o',
          #  label='E/N - E_0',
           label = f'{L=}',
           )
  axfE.plot(T_list,
          fE/L**2,
          'o',
          # label = r'$(\langle E^2 \rangle - \langle E \rangle^2)$',
          label = f'{L=}'
          )
  
  axE.legend()
  axfE.legend()
  figE.savefig(fname=f'{resultsfolder}E.pdf')
  figfE.savefig(fname=f'{resultsfolder}fE.pdf')