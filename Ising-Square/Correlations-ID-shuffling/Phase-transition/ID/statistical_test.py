import matplotlib.pyplot as plt
from dadapy import Hamming
import os,sys
import numpy as np

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
np.set_printoptions(precision=7)
# markers = ['p','p','o','o','x','x']
markers = ['s']

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)
figs,axs = plt.subplots(1)

eps = 1E-7
crossed_distances = 0

N_list = [4096]

alphamin = 1E-3
print(f'{alphamin=}')
T_list = [2.1,2.2,2.3,2.4,2.5]
# T_list = np.arange(2.2,2.6+eps,.1)
alphamax_list = np.arange(.1,.3+eps,.1)
scale_id = int(sys.argv[1])
Nr = 200
r_id_list = range(1,Nr+1)

### ID0
optfolder0 = '../ID0/results/opt/'
for N_id,N in enumerate(N_list):
  L = int(np.round(np.sqrt(N))) ; k = L
  sigma_T = np.empty(shape=(len(T_list),
                            len(alphamax_list))
                    )
  for T_id,T in enumerate(T_list):
    print(f'{T=:.2f}')
    rmaxs = np.empty(shape=(len(alphamax_list)))
    sigmas = np.empty(shape=rmaxs.shape)
    alphas = np.empty(shape=rmaxs.shape)
    logKLs = np.empty(shape=rmaxs.shape)
    for alphamax_id,alphamax in enumerate(alphamax_list):
      opt0file = optfolder0 + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
      opt0file = optfolder0 + f'L{N}_k{k}_T{T:.2f}_alphamin{alphamin:.5f}_alphamax{alphamax:.5f}.txt'
      A0 = np.loadtxt(opt0file,delimiter=',')
      # rmaxs[alphamax_id]  = A0[0]
      sigma_T[T_id,alphamax_id]  = A0[1]
      # alphas[alphamax_id]  = A0[2]
      logKLs[alphamax_id]  = A0[3]  
  axs.plot(T_list,
           sigma_T[:,scale_id]/N,
           'o-',
           )
  
### SHUFFLED ID

A = np.zeros(shape=(len(T_list),
                    len(alphamax_list),
                    4)
            )
A2 = np.zeros(shape=A.shape)

### ID-shuffled
for N_id,N in enumerate(N_list):
  for T_id,T in enumerate(T_list):
    for alphamax_id,alphamax in enumerate(alphamax_list):
      for r_id in r_id_list:
        L = int(np.round(np.sqrt(N))) ; k = L
        optfolder = f'results/opt/L{N}/k{k}/T{T:.2f}/alphamin{alphamin:.5f}/alphamax{alphamax:.5f}/'
        optfile_r = optfolder + f'rid{r_id}.txt'
        A_r = np.loadtxt(optfile_r,delimiter=',') # rmax,sigma,alpha,logKL
        A[T_id,alphamax_id,:] += A_r
        A2[T_id,alphamax_id,:] += A_r**2
      A[T_id,alphamax_id,:] /= Nr
      A2[T_id,alphamax_id,:] = np.sqrt(A2[T_id,alphamax_id,:]/Nr - A[T_id,alphamax_id,:]**2)
  print(f'{A[:,scale_id,3]=}')
  print(f'{A2[:,scale_id,3]=}')
  axs.plot(T_list,
           A[:,scale_id,1]/N,
           'o-')
  # axs.fill_between(T_list,
  #                  A[:,scale_id,1]-A2[:,scale_id,1],
  #                  A[:,scale_id,1]+A2[:,scale_id,1],
  #                  )
axs.set_ylabel(r'$d/N$')
axs.set_xlabel(r'$T$')


ax2 = axs.twinx()

ax2.plot(T_list,
         (A[:,scale_id,1]-sigma_T[:,scale_id])/N,
         color='black',
         marker='x',
         )
ax2.fill_between(T_list,
                -3*A2[:,scale_id,1]/N,
                #  0,
                +3*A2[:,scale_id,1]/N,
                color='black',
                alpha=.3)
ax2.hlines(0,T_list[0],T_list[-1],color='black',linestyles='dashed')
print(f'{sigma_T[:,scale_id]=}')
print(f'{A[:,scale_id,1]=}')
print(f'{A2[:,scale_id,1]=}')

ax2.set_ylabel(r'$(\mu - d)/N$')

axs.set_ylim(-0.1,.6)
figs.savefig(figsfolder + f'ID.png')