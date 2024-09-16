import sys
sys.path.append('LLM/')
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
from dadapy._utils.stochastic_minimization_hamming import *



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
np.set_printoptions(precision=3)
# markers = ['p','p','o','o','x','x']
markers = ['s']



figsfolder = f'results/{corpus}/{LLM}/figs/length_dependence/Nbits{Nbits}/'
if batch_randomize:
  figsfolder += f'Lconcat{Lconcat}/'
os.makedirs(figsfolder,exist_ok=True)


eps = 1E-7
crossed_distances = 0
log_scale = 0
# alphamax_list = [.05,.1,.15,.2,.25,.3,.35,.4,.45]
# alphamax_list = np.linspace(.15,.7+eps,15)[::2]
# alphamax_list = np.flip(np.arange(.45,.9+eps,.05))[::2]
alphamax_list = np.array([.7])
alphamin = 1E-5
seed = 111
delta = 7E-4
Nsteps = int(5E6)
layer_ids = [0,24]
for layer_id in layer_ids:
  print(f'{layer_id=}')
  if layer_id == 24 and LLM == 'OPT':
    emb_dim = 512
  else:
    emb_dim = 1024
  sub_lengths = np.array([i * 10  for i in range(8,30+1,2)])
  N_list = (sub_lengths) * emb_dim * Nbits
  fig,(axs,axa,axKL,axr) = plt.subplots(nrows=4,
                                        ncols=1,
                                        figsize=(10,10)
                          )

  sigmas = np.empty(shape=(len(sub_lengths),
                          len(alphamax_list)
                          )
                    )
  alphas = np.empty(shape=sigmas.shape)
  logKLs = np.empty(shape=sigmas.shape)
  rmaxs = np.empty(shape=sigmas.shape)

  for alphamax_id,alphamax in enumerate(alphamax_list):
    print(f'{alphamax=:.5f}')
    for sub_length_id,sub_length in enumerate(sub_lengths): 
      B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder0+f'sublength{sub_length}/layer_id{layer_id}/',
            )
      (rmaxs[sub_length_id,alphamax_id],
      sigmas[sub_length_id,alphamax_id],
      alphas[sub_length_id,alphamax_id],
      logKLs[sub_length_id,alphamax_id]) = B.load_results()
    output_filename = optfolder0 + f'length_dependence_l_id{layer_id}_alphamax{alphamax:.5f}_alphamin{alphamin:.5f}.txt'
    np.savetxt(fname=output_filename,
              X=np.transpose([sub_lengths,
                              sigmas[:,alphamax_id]/N_list,
                              alphas[:,alphamax_id],
                              logKLs[:,alphamax_id],
                              rmaxs[:,alphamax_id]/N_list]
                              )
                )
    
  for aux, alphamax_id in enumerate(range(len(alphamax_list))):
    alphamax = alphamax_list[alphamax_id]
    axs.plot(sub_lengths,sigmas[:,alphamax_id]/N_list,'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axa.plot(sub_lengths,alphas[:,alphamax_id],'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axKL.plot(sub_lengths,logKLs[:,alphamax_id],'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axr.plot(sub_lengths,rmaxs[:,alphamax_id]/N_list,'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
  axs.set_ylabel(r'BID/N')
  axa.set_ylabel(r'$\alpha$')
  axKL.set_ylabel(r'$\log{KL}$')
  axr.set_ylabel(r'$r*/N$')
  axr.set_xlabel('number of tokens (T)')
  # axr.set_xlabel(r'$N_{tokens}$')

  title = f'{corpus} layer{layer_id}'
  axs.set_title(title)
  box = axs.get_position()
  axs.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  fig.savefig(figsfolder + f'length_dependence_rand{randomize}_l_id{layer_id}_alphamin{alphamin:.5f}.pdf',
              bbox_inches='tight')

        

        