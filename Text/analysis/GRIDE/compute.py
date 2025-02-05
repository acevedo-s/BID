import sys,os
sys.path.append('../../../')
sys.path.append('../')
from paths import *
from dadapy import data
import numpy as np
from time import time 
from LLM.parameters import *
from LLM.LLM_paths import *
from LLM.utils import *


start = time()

layer_id = int(sys.argv[6])
print(f'{layer_id=}')
assert layer_id in layer_ids
sub_length = int(sys.argv[7])
print(f'{sub_length=}')
N_batches = int(os.getenv('N_batches'))

resultsfolder = makefolder(base=f'results/',
                          create_folder=True,
                          #  precision=2,
                          batch_randomize=batch_randomize,
                          Ns=N_batches*batch_size,
                          layer_id=layer_id,
                          sub_length=sub_length,
                          )

### ACTIVATIONS
act_outputfolder0 = get_act_outputfolder0(max_length,corpus,LLM,randomize,Lconcat,batch_randomize)
act_outputfolder = act_outputfolder0 + f'sub_length{sublength_cutoff:d}/'
X = load_activations(N_batches,
                     act_outputfolder,
                     layer_id,
                     sub_length=sub_length,
                     ).numpy()
Ns0 = X.shape[0]
X = formatting_activations(X,sub_length,Ns0,layer_normalize=0)
X = np.round(np.sign(X))
X[np.where(np.isclose(X,0))] = -1 # just in case

### GRIDE
_data = data.Data(coordinates=X, maxk=X.shape[0]-1)
range_max = X.shape[0] - 1
ids_gride, ids_err_gride, rs_gride = _data.return_id_scaling_gride(range_max=range_max)

### EXPORTING
filename='gride.txt'
np.savetxt(resultsfolder+filename,np.transpose([ids_gride,
                                                ids_err_gride,
                                                rs_gride]))

print(f'this took {(time()-start)/60:.1f} mins')