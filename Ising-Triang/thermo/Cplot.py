import numpy as np
import matplotlib.pyplot as plt
import os


rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['ggplot']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8']['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
#np.set_printoptions(precision=None)
markers = ['p','o','h','^','s']
plot_id = 0


resultsfolder = 'results/correlations/'
figsfolder = f'results/'
os.makedirs(resultsfolder,exist_ok=True)
eps = 1E-5

L_list = [30]#,40,60,100]
figC,axC = plt.subplots(1)

start = 0
T_list = np.array([.1])
i0 = 0
j0 = 0
Nsmoothing = 5 + 1 
for L_id,L in enumerate(L_list):
  for T_id,T in enumerate(T_list):
    if True:
      C = np.empty(shape=(Nsmoothing,L//2,L//2))
      for idx in range(Nsmoothing):
        C[idx,:,:] = np.loadtxt(resultsfolder + f'Corr_i0_{idx}_j0_{idx}_T{T:.2f}_L{L}.txt',)
      C = np.mean(C,axis=0)
      axC.plot(range(L//2),
              # np.abs(C),
              C[0,:],
              '-o',
              # label = f'{T=:.2f}',
              )
      axC.plot(range(L//2),
              # np.abs(C),
              C[:,0],
              '-o',
              # label = f'{T=:.2f}',
              )
axC.set_ylabel(r"$\langle \sigma_{0} \sigma_{j}\rangle$")
axC.set_xlabel("j")
axC.hlines(0,0,L,color='black')

# box = axC.get_position()
# axC.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# axC.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# axC.legend()
figC.savefig(fname=f'{figsfolder}Corr.pdf',
             bbox_inches='tight')
#   figfE.savefig(fname=f'{resultsfolder}fE.pdf')