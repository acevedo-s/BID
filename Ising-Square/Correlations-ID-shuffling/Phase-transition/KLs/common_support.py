from params import *
import matplotlib.pyplot as plt
from dadapy import Hamming
from _utils import *

#for fancy plotting
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['font.size'] = 18
plt.rcParams.update({'figure.autolayout': True})
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = plt.style.library['Solarize_Light2']['axes.prop_cycle'].by_key()['color']
from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

figsfolder = f'results/figs/'
os.makedirs(rfolder0,exist_ok=True)
os.makedirs(figsfolder,exist_ok=True)
plot = 1
quantile = 0

if quantile:
  alpha = .1

if plot:
  figh,axh = plt.subplots(1,)

k = int(round(np.sqrt(N)))
for T_id,T in enumerate(T_list):
  print(f'{T=:.2f}')
  H0 = Hamming(crossed_distances=crossed_distances)
  H0.D_histogram(
  T=T,
  L=N,
  Ns=Ns,
  resultsfolder=histfolder0,
  )
  H0.D_values,H0.D_probs = regularize_hists(N,H0.D_values,H0.D_probs)
  # print(f'{H0.D_values[0]=},{H0.D_values[-1]=}')
  # H0.remove_D_outliers()
  H0.compute_moments()
  if plot:
    axh.plot(H0.D_values,H0.D_probs,color='black')
  rs =  H0.D_values
  if quantile:
    H0.set_r_quantile(alpha)
  for r_id in range(1,Nr+1):
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
    T=T,
    L=N,
    Ns=Ns,
    resultsfolder=histfolder,
    r_id=r_id
    )
    # H.remove_D_outliers()
    H.D_values,H.D_probs = regularize_hists(N,H.D_values,H.D_probs)
    if plot:
      axh.plot(H.D_values,H.D_probs)
    # print(f'{H.D_values[0]=},{H.D_values[-1]=}')
    rs = np.intersect1d(rs,H.D_values)
  assert np.unique(np.diff(rs))[0] == 1
  rmin = rs[0]
  if quantile:
    rmax = H0.r
  else:
    rmax = rs[-1]
  np.savetxt(rfolder0 + f'rs{T:.2f}.txt',
             np.transpose([rmin,
                          rmax]),fmt='%d')


if plot:
  figh.savefig(figsfolder + 'hists.png')
