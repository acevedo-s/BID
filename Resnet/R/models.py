# import os,sys
# path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(path)
# from resnet import *

model_list = [['resnet18','ResNet18_Weights.DEFAULT'],
              # ['resnet34','ResNet34_Weights.DEFAULT'],
              # ['resnet50','ResNet50_Weights.DEFAULT'],
              # ['resnet101','ResNet101_Weights.DEFAULT'],
              # ['resnet152','ResNet152_Weights.DEFAULT']
              ]

layers_dict = {}
layers_dict['resnet18'] = [
                          # 'x', # input 
                          # 'relu', # one of these two can be added or not...
                          # 'maxpool',
                          'layer1.0.relu_1',
                          'layer1.1.relu_1',
                          'layer2.0.relu_1',
                          'layer2.1.relu_1',
                          'layer3.0.relu_1',
                          'layer3.1.relu_1',
                          'layer4.0.relu_1',
                          'layer4.1.relu_1',
                          'flatten',
                          ]

layers_dict['resnet152'] = ['x',
                           'layer1.2.relu_2',
                           'layer2.7.relu_2',
                           'layer3.10.relu_2',
                           'layer3.20.relu_2',
                           'layer3.35.relu_2',
                           'layer4.0.relu_2',
                           'layer4.1.relu_2',
                           'layer4.2.relu_2',
                           'flatten',
                           ]