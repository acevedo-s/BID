import numpy as np
from time import sleep
eps = 1E-6

# T_list = np.arange(.71,.75+eps,.002)
### low temperature phase
T_list = np.arange(.5,.7+eps,0.1)
T_list = np.concatenate((T_list,
                         np.arange(.71,.74+eps,.002))
                         )

### high temperature phase
T_list = np.concatenate((T_list,
                         np.arange(.75,.77+eps,.002))
                         )
T_list = np.concatenate((T_list,
                         np.arange(.8,1+eps,.1))
                         )
np.savetxt(fname=f'T_list{len(T_list)}.txt',X=T_list,fmt='%.3f')
