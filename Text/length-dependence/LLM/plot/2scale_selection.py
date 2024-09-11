import sys,os
sys.path.append('LLM/')
from parameters import *
import matplotlib.pyplot as plt
import numpy as np
os.environ["JAX_ENABLE_X64"] = "True"
from dadapy import Hamming
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
np.set_printoptions(precision=7)
markers = ['s']
figsfolder = f'results/{corpus}/{LLM}/figs/2BID/model_validation/'
# resultsfolder = f'results/{corpus}/{LLM}/selected_BIDs/'
if batch_randomize:
  figsfolder += f'Lconcat{Lconcat}/'
  # resultsfolder += f'Lconcat{Lconcat}/'
os.makedirs(figsfolder,exist_ok=True)
# os.makedirs(resultsfolder,exist_ok=True)

tau_list = [9,59,109,159,209,259]
tau_list [9]
eps = 1E-7

seed = 111
delta = 7E-4
Nsteps = int(5E6)

# alphamax_list = [.01,.05,.1,.15,.2,.25,.3,.35,.4,.45]
# alphamax_list = np.linspace(.15,.7,15)[-1:]
# alphamax_list = np.arange(.75,.9+eps,.05)[-1:]
alphamin_list = [1E-5]#,1E-4,1E-3,1E-2,1E-1]
alphamax_list = [.2,.5]

# sigmas = np.empty(shape=(len(tau_list),
#                          len(alphamax_list)
#                          )
#                   )
# alphas = np.empty(shape=sigmas.shape)
# logKLs = np.empty(shape=sigmas.shape)
# rmaxs = np.empty(shape=sigmas.shape)

layers = [24]
d0s = np.empty(shape=len(layers),dtype=float)
logKLs = np.empty(shape=d0s.shape,dtype=float)

for sub_length_id,sub_length in enumerate(tau_list):
  print(f'{sub_length=}')
  for layer_aux_id,layer_id in enumerate(layers):
    print(f'{layer_id=}')
    figh,axh = plt.subplots(1)
    logKLs_aux = np.empty(shape=(
                          len(alphamax_list),
                          len(alphamin_list)),
                          dtype=float)
    d0s_aux = np.empty(shape=logKLs_aux.shape,dtype=float)
    for alphamax_id,alphamax in enumerate(alphamax_list):
      for alphamin_id,alphamin in enumerate(alphamin_list):
        if layer_id == 24 and LLM == 'OPT':
          emb_dim = 512
        else:
          emb_dim = 1024
        N = emb_dim * 2
        B = BID(
              alphamin=alphamin,
              alphamax=alphamax,
              seed=seed,
              delta=delta,
              Nsteps=Nsteps,
              optfolder0=optfolder0 + f'layer{layer_id}/tau{tau}/',
              )
        _,d0s_aux[alphamax_id,alphamin_id],_,logKLs_aux[alphamax_id,alphamin_id] = B.load_results()
    best_ids = np.where(np.isclose(logKLs_aux,np.nanmin(logKLs_aux)))
    # print(f'{best_ids=}')
    # print(f'{logKLs_aux[best_ids][0]=}')
    logKLs[layer_aux_id] = logKLs_aux[best_ids][0]
    d0s[layer_aux_id] = d0s_aux[best_ids][0] / N
    # print(f'{logKLs[layer_aux_id]=}')
    # print(f'{d0s[layer_aux_id]=}')
    B.alphamax = alphamax_list[best_ids[0][0]]
    B.alphamin = alphamin_list[best_ids[1][0]]
    print(f'{B.alphamin=:.5f}',f'{B.alphamax=:.2f}')
    (remp,Pemp,Pmodel) = B.load_fit()

    axh.plot(remp,
            Pemp,
            'x',
            label=f'{sub_length=}',
            zorder=0,
            )
    axh.plot(remp,
            Pmodel,
            zorder=1,
            color='black',
            )
    axh.set_xlabel(r'$r$')
    axh.set_ylabel('P(r)')
    axh.set_yscale('log')

    box = axh.get_position()
    axh.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axh.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    figh.savefig(figsfolder + f'tau{tau}_l{layer_id}_alphamin{B.alphamin:.5f}_alphamax{B.alphamax:.5f}.pdf',bbox_inches='tight')      
    plt.close()
  # np.savetxt(resultsfolder + f'sublength{sub_length}.txt',
  #           np.column_stack((np.array(layers),
  #                           d0s,
  #                           logKLs)
  #                           )
  #           )