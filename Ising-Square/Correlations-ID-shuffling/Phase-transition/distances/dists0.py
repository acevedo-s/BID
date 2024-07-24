from params import *
from utils import *

for T_id, T in enumerate(T_list):
  start = time()
  print(f'T={T:.2f}')
  fname = (datafolder0 + f'L{L}_seed_{seed}/T{T:.2f}.npy')
  xy = np.load(fname)
  if Z2flag:
    xy = fix_Z2symmetry(xy)
  elif Mplusflag:
    xy = fix_Mpositive(xy)
  print(f'{xy.shape=}')
  # print(xy.sum()/L)
  Ns = xy.shape[0]
  H = Hamming(coordinates=xy,
              crossed_distances=crossed_distances)
  H.compute_distances()
  H.D_histogram(compute_flag=1,
                save=True,
                T=T,
                L=L,
                Ns=Ns,
                resultsfolder=histfolder,
                )
  print(f'{T=:.2f} took {(time()-start)/60.:.1f} mins')
