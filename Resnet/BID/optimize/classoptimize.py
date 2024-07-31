import numpy as np
import sys
sys.path.append('../../')
from parameters import * 
from dadapy import Hamming
from dadapy._utils.stochastic_minimization_hamming import *

alphamin = float(sys.argv[4])
alphamax = float(sys.argv[5])

delta = 5E-4
Nsteps = int(1E6)
seed = 1

### to get the number of spins, for the initial condition of the optimization
Ns,N = np.genfromtxt(EDfile,
                    dtype='str',
                    unpack=True).astype(int)
print(f'{N=},{Ns=}')

H = Hamming()
H.D_histogram(Ns=Ns,
              resultsfolder=histfolder)
H.compute_moments()
# print(f'{H.D_values=}')
# print(f'{H.D_mu_emp=}')
# print(f'{H.D_var_emp=}')

B = BID(H,
        alphamin=alphamin,
        alphamax=alphamax,
        seed=seed,
        delta=delta,
        Nsteps=Nsteps,
        optfolder0=optfolder0,
        export_logKLs=0,
        L=N
        )
B.computeBID()




# print(f'{layer_name}')
# optfolder = optfolder0 + f'{key}/{layer_name}/alphamin{alphamin:.5f}/'
# optfolder += f'delta{delta}/seed{seed}/'
# optfile = optfolder + 'opt.txt'
# os.makedirs(optfolder,exist_ok=True)
# H = Hamming()
# H.D_histogram(
#   Ns=Ns,
#   resultsfolder=histfolder+f'{key}/{layer_name}/',
#   )
# H.compute_moments()
# H.set_r_quantile(alphamin)
# rmin = H.r
# idmin = H.r_idx
# H.r = None
# H.r_idx = None
# KL_list = []
# sigma_list = []
# alpha_list = []
# rmaxs = []
# for alphamax_id,alphamax in enumerate(alphamax_list):
#   H.set_r_quantile(alphamax)
#   rmax = H.r
#   idmax = H.r_idx
#   H.r = None
#   H.r_idx = None
#   remp = jnp.array(H.D_values[idmin:idmax+1], dtype=jnp.float64)
#   Pemp = jnp.array(H.D_probs[idmin:idmax+1], dtype=jnp.float64)
#   Pemp /= jnp.sum(Pemp)
#   Pmodel = jnp.zeros(shape=Pemp.shape, dtype=jnp.float64)
#   key0 = random.PRNGKey(seed)
#   sigma0 = H.D_mu_emp
#   alpha0 = alpha00
#   ### First layer has problems when crop_size is very small:
#   if layer_id <= 3 and crop_size <= 100:
#     help_id = layer_id
#     helpfolder = helpfolder0 + f'{key}/{layer_names[help_id]}/alphamin{alphamin:.5f}/'
#     helpfolder += f'delta{delta}/seed{seed}/'
#     helpfile= helpfolder + 'opt.txt'
#     A = np.loadtxt(helpfile,delimiter=',')
#     if len(A.shape)==1:
#       A = np.vstack([A,A])
#     helpsigmas = A[:,1]
#     helpalphas = A[:,2]
#     helplogKLs = A[:,3]
#     helpminKL_id = np.where(np.isclose(helplogKLs,np.nanmin(helplogKLs)))[0][0]
#     sigma0 = helpsigmas[helpminKL_id]
#     alpha0 = helpalphas[helpminKL_id]
#   Op = Optimizer(key=key0,
#                 sigma=jnp.double(sigma0),
#                 alpha=jnp.double(alpha0),
#                 delta=jnp.double(delta),
#                 remp=remp,
#                 Pemp=Pemp,
#                 Pmodel=Pmodel,
#                 Nsteps=Nsteps,
#                 )
#   nan_counts,inf = check_initial_condition(Op)
#   if nan_counts != 0: 
#     print('Pmodel gives nan for the initial conditions')
#     continue
#   if inf != 0: 
#     print('KL is infinite for the initial conditions')
#     continue
#   Op = minimize_KL(Op)
#   print(f'{alphamax=:.2f},{Op.sigma:.8f},{Op.alpha:8f},{jnp.log(Op.KL)=}')
#   if alphamax_id == 0:
#     if export_output_flag:
#       if remove_previous_output:
#         os.system(f'rm -f {optfile}')
#   print(f'{rmax:d},{Op.sigma:.8f},{Op.alpha:8f},{np.log(Op.KL):.8f},{alphamax:.5f}',
#         file=open(optfile, 'a'))
#   print(f'{Op.acc_ratio=}')
#   print(f'this took {(time()-start)/60.:.1f} mins')
#   # ### using previous solution as starting point:
#   # alpha0 = Op.alpha
#   # sigma0 = Op.sigma