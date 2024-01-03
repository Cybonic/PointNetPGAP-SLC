import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.pooling import *
from ..backbones.pointnet import *
from ..backbones import resnet
from networks.utils import *

class PointNetGeM(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024,p=3):
        super(PointNetGeM, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.head = GeM(outdim=output_dim, p=p, eps=1e-6)

    def forward(self, x):
        x = self.point_net(x)
        x = self.head(x)
        return x

  
    def __str__(self):
        return "PointNetGeM"


class ResNet50GeM(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, output_dim=1024):
        super(ResNet50GeM, self).__init__()

        return_layers = {'layer4': 'out'}
        param = {'pretrained_backbone': False,
                    'out_dim': output_dim,
                    'feat_dim': feat_dim,
                    'in_channels': in_dim,
                    'max_points': num_points,
                    'modality': 'bev'} 
         
        #pretrained = resnet50['pretrained_backbone']

        #max_points = model_param['max_points']
        backbone = resnet.__dict__['resnet50'](param)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        self.head = GeM(outdim=output_dim, p=3, eps=1e-6)

    def forward(self, x):
        x = self.backbone(x)
        x = x['out']
        x = self.head(x)
        return x

  
    def __str__(self):
        return "ResNet50GeM"
    

