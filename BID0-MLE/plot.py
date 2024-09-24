import os,sys
import numpy as np
import matplotlib.pyplot as plt

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


np.set_printoptions(precision=7)
figsfolder = f'results/figs/'
resultsfolder = f'results/BID/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7
Ns = 5000

L = 100#[int(sys.argv[1])
N = L**2
T_list = [2.3,2.4,2.5]
# T_list = [2.0,2.1,2.2]
fig,ax = plt.subplots(1)
figh,axh = plt.subplots(1)

for T_id,T in enumerate(T_list):
  rs,ds,probs = np.loadtxt(resultsfolder + f'T{T:.2f}.txt',unpack=True)
  ax.plot(rs/N,ds/N,label=f'{T=:.2f}')
  axh.plot(rs/N,probs,label=f'{T=:.2f}')
ax.legend()
ax.set_xlabel(r'$r/N$')
ax.set_ylabel(r'$\hat{d}(r)/N$')

fig.savefig(figsfolder + f'scale_dependence_{T_list}.pdf')
figh.savefig(figsfolder + f'hists.pdf')

