import numpy as np
import sys,os
import matplotlib.pyplot as plt

figsfolder = f'results/figs/'
os.makedirs(figsfolder,exist_ok=True)

geometry = 'Ising-square'
eps = 1E-7
metric = 'hamming'
crossed_distances = 0
L = 100
T = float(sys.argv[1])
N = L**2

datafolder = f'/scratch/sacevedo/{geometry}/canonical/'
datafile = datafolder + f'L{L}/T{T:.2f}.txt'
X_id = 4321
if T < np.inf:
  X = np.loadtxt(f'{datafile}').astype(int)[X_id].reshape((L,L))
else:
  X = 2*np.random.randint(low=0,high=2,size=(N))-1
  X.reshape((L,L))

fig,ax = plt.subplots(1)
plt.tick_params(axis='both', 
                which='both', 
                bottom=False, 
                top=False, 
                left=False, 
                right=False, 
                labelbottom=False, 
                labelleft=False)
ax.imshow(X,
          cmap='gray',
          vmin=-1,
          vmax=1)
fig.savefig(figsfolder + f'X_T{T:.2f}_{X_id}.pdf',
            bbox_inches='tight')

