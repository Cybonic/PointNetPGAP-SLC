#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
#from .heads.netvlad import NetVLADLoupe
from networks.utils import *
from networks.backbones import resnet, pointnet
from networks.backbones.spvnas.model_zoo import spvcnn

from networks.aggregators import multihead 
import yaml
import os

def PointNetORCHNetSGMaxPoolingFC(**argv):
  return ORCHNet('pointnet',aggregator='MultiHeadSGMaxPoolingFC',**argv)

def spvcnnORCHNetMeanSOP(**argv):
  return ORCHNet('spvcnn',aggregator='MeanSOP',**argv)

def spvcnnORCHNet(**argv):
  return ORCHNet('spvcnn',**argv)

def PointNetORCHNet(**argv):
  return ORCHNet('pointnet',**argv)

def ResNet50ORCHNet(**argv):
  return ORCHNet('resnet50',**argv)

def spvcnnORCHNetSPoC(**argv):
  return ORCHNet('spvcnn',aggregator='SPoC',**argv)

def ResNet50ORCHNetMaxPooling(**argv):
  return ORCHNet('resnet50',aggregator='MultiHeadMAXPolling',**argv)

def PointNetORCHNetMaxPooling(**argv):
  return ORCHNet('pointnet',aggregator='MultiHeadMAXPolling',**argv)

def PointNetORCHNetSOP(**argv):
  return ORCHNet('pointnet',aggregator='SOP',**argv)

def PointNetORCHNetVLADSPoCMaxPooling(**argv):
  return ORCHNet('pointnet',aggregator='VLADSPoCMaxPooling',**argv)

def PointNetORCHNetVLADSPoCLearned(**argv):
  return ORCHNet('pointnet',aggregator='VLADSPoCLearned',**argv)

class ORCHNet(nn.Module):
  def __init__(self,backbone_name:str,output_dim:int,feat_dim:int,aggregator="MultiHead",**argv):
    super(ORCHNet,self).__init__()
    
    #model_cfg = os.path.join('networks',f'{modelname.lower()}','param.yaml')
    dirname, filename = os.path.split(os.path.abspath(__file__))
    model_cfg_file = os.path.join(dirname,'parameters','orchnet.yaml')
    model_cfg = yaml.safe_load(open(model_cfg_file, 'r'))
  
    assert backbone_name in model_cfg,'Backbone param do not exist'
    print("Opening model config file: %s" % model_cfg_file)
    model_param = model_cfg[backbone_name]

    self.backbone_name = backbone_name
    if self.backbone_name == 'resnet50':
      if 'num_points' in argv:
        argv.pop('num_points')
      return_layers = {'layer4': 'out'}
      pretrained = model_param['pretrained_backbone']
      #max_points = model_param['max_points']
      backbone = resnet.__dict__[backbone_name](pretrained,**argv)
      self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    elif self.backbone_name == 'pointnet':
      self.backbone = pointnet.PointNet_features(dim_k=feat_dim,use_tnet=False,scale=1)

    elif self.backbone_name == 'spvcnn':
      self.backbone = spvcnn(output_dim=16)

    # Define Aggregator
    self.aggregator = aggregator
    if aggregator.endswith('SOP'):
      import networks.aggregators.SOP as SOP
      self.head = SOP.__dict__[aggregator](input_dim=16,is_tuple=False)
    else:
      assert aggregator in multihead.__dict__,'Aggregator param do not exist'
      self.head  = multihead.__dict__[aggregator](outdim=output_dim)

   
  def forward(self,x):

    y = self.backbone(x)
    if self.backbone_name == 'resnet50':
      y = y['out']
    elif self.backbone_name == 'spvcnn':
      _, counts = torch.unique(x.C[:, -1], return_counts=True)
      y = torch.split(y, list(counts))
      y = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
      if self.aggregator != 'SOP':
        # swap axis
        y = y.permute(0,2,1)
      #width = int(np.sqrt(y.shape[2]))
      #y = y.view(y.size(0),y.size(1),width,width) 
         
    z = self.head(y)

    return z
  
  def get_backbone_params(self):
    return self.backbone.parameters()

  def get_classifier_params(self):
    return self.head.parameters()
  
  def __str__(self):
    return self.backbone_name + "-ORCHNet-" + self.aggregator


if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir)
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)
    lidar_pc = lidar_pc.reshape(-1, 4)
    input = torch.tensor(lidar_pc[:,:3]).cuda()
    input = input.reshape(1,1,-1, 3) # B,C,N,4
    #input = make_sparse_tensor(lidar_pc, 0.05).cuda()
    batch = torch.cat((input,input))

    model = ORCHNet('pointnet').cuda()
    model.train()
    output = model(batch)
    print('output size: ', output[0].size())







