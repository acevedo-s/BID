import numpy as np
from time import sleep
eps = 1E-6

T_list = np.arange(.1,4+eps,.1)
np.savetxt(fname=f'T_list{len(T_list)}.txt',
           X=T_list,
           fmt='%.3f'
          )
sleep(1)
