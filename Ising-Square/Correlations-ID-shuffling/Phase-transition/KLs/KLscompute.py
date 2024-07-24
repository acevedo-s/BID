from params import *
from dadapy import Hamming
from _utils import *
# from scipy.stats import chisquare
from time import sleep

np.set_printoptions(precision=6)


os.system(f'python common_support.py') ; sleep(.2)

os.makedirs(KLsfolder,exist_ok=True)
meanKLs_filename = KLsfolder + f'mean_KLs.txt'
os.system(f'rm -f {meanKLs_filename}')
k = int(round(np.sqrt(N)))
KL0 = np.empty(shape=(len(T_list),
                      Nr)
              )

for T_id,T in enumerate(T_list):
  print(f'{T=:.2f}')
  KLs = []
  rmin,rmax = np.loadtxt(rfolder0 + f'rs{T:.2f}.txt',
                         dtype=int)
  shuff_probs = np.empty(shape=(Nr,rmax+1-rmin))
  H0 = Hamming(crossed_distances=crossed_distances)
  H0.D_histogram(
  T=T,
  L=N,
  Ns=Ns,
  resultsfolder=histfolder0,
  )
  H0.D_values,H0.D_probs = regularize_hists(N,H0.D_values,H0.D_probs)
  H0.D_values,H0.D_probs = set_common_support(H0.D_values,
                                              H0.D_probs,
                                              rmin,
                                              rmax,
                                              )
  for r_id in range(Nr):
    H = Hamming(crossed_distances=crossed_distances)
    H.D_histogram(
    T=T,
    L=N,
    Ns=Ns,
    resultsfolder=histfolder,
    r_id=r_id+1,
    )
    H.D_values,H.D_probs = regularize_hists(N,H.D_values,H.D_probs)
    H.D_values,H.D_probs = set_common_support(H.D_values,
                                              H.D_probs,
                                              rmin,
                                              rmax,
                                              )
    shuff_probs[r_id,:] = H.D_probs
    assert len(shuff_probs[0]) == len(H.D_probs)
    KL0[T_id,r_id] = KLd_PQ(H.D_probs,
                           H0.D_probs)
    for rr_id in range(r_id):
      KLs.append(KLd_PQ(H.D_probs,
                        shuff_probs[rr_id,:])
                )
  ### FULL KL DISTRIBUTIONS
  KLs = np.array(KLs)
  KLs_filename = KLsfolder + f'KLs_T{T:.2f}.txt'
  # np.savetxt(KLs_filename,np.transpose([H.D_values,KLs]),fmt='%s')
  np.savetxt(KLs_filename,KLs)
  KLs0_filename = KLsfolder + f'KLs0_T{T:.2f}.txt'
  np.savetxt(KLs0_filename,KL0[T_id,:])

  ### MEANS
  mean = np.mean(KLs)
  std = np.std(KLs)
  print(mean,std,file=open(meanKLs_filename,'a'))
  np.savetxt(KLsfolder + f'mean_KLs0.txt',
          np.transpose([np.mean(KL0,axis=1),np.std(KL0,axis=1)])
          )

os.system(f'python plotmeanKLs.py')