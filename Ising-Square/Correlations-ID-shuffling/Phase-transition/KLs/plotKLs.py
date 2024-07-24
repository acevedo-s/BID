from params import *
import matplotlib.pyplot as plt
from dadapy import Hamming
from scipy.stats import kstest

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

T_list = [2.5]
# T_list = T_list[::4]
ks_list = []
for T_id,T in enumerate(T_list):
  KLsfilename = KLsfolder + f'KLs_T{T:.2f}.txt'
  KLs0filename = KLsfolder + f'KLs0_T{T:.2f}.txt'
  KLs = np.loadtxt(KLsfilename,unpack=True)
  KLs0 = np.loadtxt(KLs0filename,unpack=True)
  KLs0.sort()
  KLs.sort()
  # print(len(KLs0))
  # print(len(KLs))
  indeces0 = range(1,len(KLs0)+1)
  F_emp0 = np.array(indeces0)/len(KLs0)
  indeces = range(1,len(KLs)+1)
  F_emp = np.array(indeces)/len(KLs)
  ax.plot(KLs0,F_emp0,'o-',
          # color='black',
          label=r'$KL_0$',
          )
  ax.plot(KLs,
          F_emp,
          'x-',
          label=r'$KL$',
          )
  ks = kstest(KLs,KLs0).pvalue
  ks_list.append(ks)
  print(f'{ks=}')
  color_id = 2

figT,axT = plt.subplots()
axT.plot(T_list,ks_list)
axT.vlines(2.269,axT.get_ylim()[0],axT.get_ylim()[1],linestyles='dashed',color='gray')
figT.savefig(figsfolder + 'KST.png')



if log_scale:
  # ax.set_yscale('log')
  ax.set_xscale('log')

ax.set_ylabel('F')
ax.set_xlabel('KL')
# ax.legend()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.savefig(figsfolder + 'cumulatives.png')
