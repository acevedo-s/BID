import sys,os
import matplotlib.pyplot as plt
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import BID
from scipy.optimize import curve_fit


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
# colors = plt.style.library['seaborn-v0_8-dark-palette']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=3)
markers = ['p','o','h','^','s','P','>']

plot_id = 0


figsfolder = f'results/figs/'
_optfolder = f'results/opt/'
os.makedirs(figsfolder,exist_ok=True)
eps = 1E-7

fig,ax = plt.subplots(1)
fig0,ax0 = plt.subplots(1)

L_list = np.array([30,40,50,60,70,80,90,100,110,120,130])
N_list = L_list**2
T_list = [.1]

alphamin = 0 # order of quantile for P(r)
alphamax = .05 #5E-3
Nsteps = int(1E6)
seed = 1
delta = 5E-4

sigmas = np.empty(shape=(len(L_list),
                         len(T_list)
                        )
                  )
rmaxs = np.empty(shape=sigmas.shape,dtype=int)
alphas = np.empty(shape=sigmas.shape)
logKLs = np.empty(shape=sigmas.shape)

for T_id,T in enumerate(T_list):
  for L_id,L in enumerate(L_list):
    optfolder0 = _optfolder + f'L{L}/T{T:.2f}/'
    B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder0,
            )
    (rmaxs[L_id,T_id],
    sigmas[L_id,T_id],
    alphas[L_id,T_id],
    logKLs[L_id,T_id]) = B.load_results()

d0 = sigmas[:,0]/N_list
x0N = False
if x0N:
  x0 = N_list.astype(float)**-1
else:
  x0 = L_list.astype(float)**-1
ax0.scatter(x0,
            d0,
            )
ax0.set_xlim(0,np.max(x0)*1.15)
ax0.set_ylim(0,.42)
(a,b),cov = np.polyfit(x0,d0,1,cov=True)
print(f'{a=}')
print(f'{b=}')
print(f'{cov=}')
x0 = np.linspace(ax0.get_xlim()[0],
                 ax0.get_xlim()[1])
ax0.plot(x0,
        a*x0+b,
        color='black',
        label=f'offset={b:.2f} +- {cov[1][1]:.3f}' 
        )

ax0.set_ylabel('BID(T=0)/N')

if x0N: 
  ax0.set_xlabel('1/N')
else:
  ax0.set_xlabel('1/L')

ax0.legend()
fig0.savefig(figsfolder + f'd_00_linear.pdf',
             bbox_inches='tight',
             )   