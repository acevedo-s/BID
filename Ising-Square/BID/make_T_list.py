import numpy as np
eps = 1E-6
T_list = np.arange(1.5,2.2+eps,.1)
T_list = np.concatenate((T_list,
                         np.arange(2.21,2.39+eps,.01))
                         )
T_list = np.concatenate((T_list,
                         np.arange(2.4,4+eps,.1)
                         )
                         )
np.savetxt('T_list.txt',T_list,fmt='%.3f')