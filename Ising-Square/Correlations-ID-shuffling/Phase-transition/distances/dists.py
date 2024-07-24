from params import *
from utils import *

start = time()
T = float(sys.argv[1])
print(f'T={T:.2f}')
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
fname = (datafolder0 + f'L{L}_seed_{seed}/T{T:.2f}.npy')
xy = np.load(fname).astype(int)[:,:]
if Z2flag:
  xy = fix_Z2symmetry(xy)
elif Mplusflag:
  xy = fix_Mpositive(xy)
print(f'{xy.shape=}')
Ns = xy.shape[0]
cut = int(L/2)
x=xy[:,:cut]
y=np.copy(xy[:,cut:])
shuf=np.empty([Ns,L])
np.random.shuffle(y)
shuf[:,:cut]=x
shuf[:,cut:]=y
H = Hamming(coordinates=shuf,
            crossed_distances=crossed_distances)
H.compute_distances()
H.D_histogram(compute_flag=1,
              save=True,
              r_id=task_id,
              L=L,
              Ns=Ns,
              T=T,
              resultsfolder=shuff_histfolder,
              )
print(f'{T=:.2f} took {(time()-start)/60.:.1f} mins')

