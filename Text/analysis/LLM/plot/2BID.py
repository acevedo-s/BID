import sys
sys.path.append('LLM/')
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
os.environ["JAX_ENABLE_X64"] = "True"
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



figsfolder = f'results/{corpus}/{LLM}/figs/2BID/'
if batch_randomize:
  figsfolder += f'Lconcat{Lconcat}/'
os.makedirs(figsfolder,exist_ok=True)


eps = 1E-7
log_scale = 1
# alphamax_list = [.05,.1,.15,.2,.25,.3,.35,.4,.45]
# alphamax_list = np.linspace(.15,.7+eps,15)[::2]
# alphamax_list = np.flip(np.arange(.45,.9+eps,.05))[::2]
alphamax_list = np.array([.2])
alphamin = 1E-5
seed = 111
delta = 7E-4
Nsteps = int(5E6)
layer_ids = [24]
tau_list = [9,59,109,159,209,259]
for layer_id in layer_ids:
  print(f'{layer_id=}')
  if layer_id == 24 and LLM == 'OPT':
    emb_dim = 512
  else:
    emb_dim = 1024
  N = 2* emb_dim
  fig,(axs,axa,axKL,axr) = plt.subplots(nrows=4,
                                        ncols=1,
                                        figsize=(10,10)
                          )

  sigmas = np.empty(shape=(len(tau_list),
                          len(alphamax_list)
                          )
                    )
  alphas = np.empty(shape=sigmas.shape)
  logKLs = np.empty(shape=sigmas.shape)
  rmaxs = np.empty(shape=sigmas.shape)

  for alphamax_id,alphamax in enumerate(alphamax_list):
    print(f'{alphamax=:.5f}')
    for tau_id,tau in enumerate(tau_list): 
      B = BID(
            alphamin=alphamin,
            alphamax=alphamax,
            seed=seed,
            delta=delta,
            Nsteps=Nsteps,
            optfolder0=optfolder0 + f'layer{layer_id}/tau{tau}/',
            )

      (rmaxs[tau_id,alphamax_id],
      sigmas[tau_id,alphamax_id],
      alphas[tau_id,alphamax_id],
      logKLs[tau_id,alphamax_id]) = B.load_results()
    # output_filename = optfolder0 + f'length_dependence_l_id{layer_id}_alphamax{alphamax:.5f}_alphamin{alphamin:.5f}.txt'
    # np.savetxt(fname=output_filename,
    #           X=np.transpose([sub_lengths,
    #                           sigmas[:,alphamax_id]/N_list,
    #                           alphas[:,alphamax_id],
    #                           logKLs[:,alphamax_id],
    #                           rmaxs[:,alphamax_id]/N_list]
    #                           )
    #             )
    
  for aux, alphamax_id in enumerate(range(len(alphamax_list))):
    alphamax = alphamax_list[alphamax_id]
    axs.plot(tau_list,sigmas[:,alphamax_id]/N,'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axa.plot(tau_list,alphas[:,alphamax_id],'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axKL.plot(tau_list,logKLs[:,alphamax_id],'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
    axr.plot(tau_list,rmaxs[:,alphamax_id]/N,'s-',color=colors[aux%len(colors)],label=f'a={alphamax:.5f}')
  axs.set_ylabel(r'BID/N')
  axa.set_ylabel(r'$\d_1$')
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

        

        