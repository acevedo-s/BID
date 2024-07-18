from R import *

nodes_filename = 'Rnodes.txt'
os.system(f'rm -f {nodes_filename}')
R = Resnet(model_name,W_model)
R.get_nodes()
print(R.nodes,file=open(nodes_filename,'a'))
print(layers_dict[model_name],file=open(nodes_filename,'a'))