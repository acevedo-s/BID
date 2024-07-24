from params import *
import matplotlib.pyplot as plt
from dadapy import Hamming
from _utils import KLd_PQ
from scipy.stats import chisquare

#for fancy plotting
rcpsize = 20
plt.rcParams['xtick.labelsize']= rcpsize
plt.rcParams['ytick.labelsize']=rcpsize
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = rcpsize
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['Solarize_Light2']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)
# print(plt.rcParams.keys())
np.set_printoptions(precision=6)
plot_id = 0

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)

log_scale = 1
fig,ax = plt.subplots(1,)
ax2 = ax.twinx()

KLsfilename = KLsfolder + f'mean_KLs.txt'
KLs0filename = KLsfolder + f'mean_KLs0.txt'

mu,std = np.loadtxt(KLsfilename,unpack=True)
mu0,std0 = np.loadtxt(KLs0filename,unpack=True)
print(f'{mu0=}')
ax.vlines(2.269,ax.get_ylim()[0],ax.get_ylim()[1],linestyles='dashed',color='gray')
ax.plot(T_list,mu0,'o-',color='black',label=r'$\mu_0$')
color_id = 2
ax.fill_between(T_list,
                mu0-3*std0,
                mu0+3*std0,
                alpha=.5,
                color='black',
                )
print(f'{std0=}')
ax.plot(T_list,mu,'o-',label=r'$\mu$',color=colors[color_id])
ax.fill_between(T_list,
                mu-3*std,
                mu+3*std,
                alpha=.5,
                color=colors[color_id]
                )
ax2.plot(T_list,mu0-mu)

if log_scale:
  ax.set_yscale('log')

ax.set_ylabel('KL')
ax.set_xlabel('T')
ax.legend()

fig.savefig(figsfolder + 'mean_KLs.png')
