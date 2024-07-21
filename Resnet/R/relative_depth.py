from .resnet import *
from .models import *

relative_depth_dict = {}
for model_id in range(len(model_list)):
  model_name,W_model = model_list[model_id]
  layer_names = layers_dict[model_name]
  R = Resnet(model_name,W_model)
  R.get_nodes()
  relative_depth = []
  for layer_name_id, layer_name in enumerate(layer_names):
    relative_depth.append(np.where(R.nodes==layer_name)[0][0])
  relative_depth = np.array(relative_depth) / len(R.nodes)
  relative_depth_dict[model_name]  = relative_depth
