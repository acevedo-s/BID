import torch
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import numpy as np
import os
import pickle

class Resnet():
  def __init__(self,model_name,
               model_weights,
               ):
    self.model_name = model_name
    self.model_weights = model_weights
    self.load_model()

  def load_model(self):
    self.model = torch.hub.load('pytorch/vision:v0.10.0',
                            self.model_name, 
                            weights=self.model_weights
    )
    self.model.eval() # evaluation mode

  def get_nodes(self):
    """ 
    This routine extracts the node names in the model.
    See 
    https://pytorch.org/vision/stable/generated/torchvision.models.feature_extraction.get_graph_node_names.html
    """
    self.nodes = np.array(get_graph_node_names(self.model)[1]) # returning only evaluation nodes
    return
  
  def export_features(self,file,obj):
    """
    
    """
    with open(file,'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return
  
  def extract_features_from_layer(self,
                                  data,
                                  layer_name,
                                  resultsfolder,
                                  i0,
                                  chunk_size
                                  ):
    """
    *Inputs: 
    - input_data (batch_size,3,crop_size,crop_size) Torch floating tensor # default crop_size is 224
    - layer name (str): the name of a single layer, see *Notes for format 
    - model: trained_model
    - resultsfolder: path to where to save extracted_features

    *Notes: 
    - code snippet to see node names: 
    from torchvision.models.feature_extraction import get_graph_node_names
    nodes, _ = get_graph_node_names(model)
    print(nodes)
    - the variable <out> is a dictionary with keys like 'x', 'flatten' and 
      values given by the corresponding features
    """
    
    # using torch feature extractor:
    feature_extractor = create_feature_extractor(
	                      self.model,
                        return_nodes = [layer_name]
    )
    out = feature_extractor(data)
    file = resultsfolder + layer_name + '/'
    os.makedirs(file,exist_ok=True)
    file += f'features_dict_i0_{i0}_chunk_size_{chunk_size}.pickle'
    self.export_features(file,out)
    return
  

def load_features_layer(act_folder,
                        layer_name,
                        chunk_size,
                        i0=0,
                        i_max=None,
                        flatten=False,
                        ):

    """
    i0: initial integer value for chunk index
    chunk_size: how many input images to process in parallel
    i_max: maximum number of chunks to load

    """
    if i_max == None: i_max = i0
    
    # defining path:
    file0 = act_folder + layer_name + '/'
    file = file0 + f'features_dict_i0_{i0}_chunk_size_{chunk_size}.pickle'

    # first chunk
    with open(file, 'rb') as handle:
      aux = pickle.load(handle)[layer_name]
    i0 += 1

    # the rest:
    while (i0 < i_max):
      file = file0 + f'features_dict_i0_{i0}_chunk_size_{chunk_size}.pickle'
      with open(file, 'rb') as handle:
        features = pickle.load(handle)[layer_name]
      aux = torch.cat((aux,features),dim=0)
      i0 += 1
    if flatten:
      aux = torch.flatten(aux,start_dim=1)
      np.savetxt(file0+f'a_flatten_shape.txt',aux.shape,fmt='%d')
    else:
      np.savetxt(file0+f'a_shape.txt',aux.shape,fmt='%d')
    print(f'shape of features:{aux.shape}')
    
    return aux.detach().numpy()
