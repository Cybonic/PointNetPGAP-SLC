import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.pooling import *
from ..backbones.pointnet import *
from ..backbones import resnet
from networks.utils import *
from networks.backbones.spvnas.model_zoo import spvcnn
from networks.pipelines.pipeline_utils import *

class PointNetMAC(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024,**argv):
        super(PointNetMAC, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        self.head = MAC(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x)
        x = self.head(x)
        return x
    
    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.head.parameters()
  
    def __str__(self):
        return "PointNetMAC"


class ResNet50MAC(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, output_dim=1024,**argv):
        super(ResNet50MAC, self).__init__()

        return_layers = {'layer4': 'out'}
        param = {'pretrained_backbone': False,
                    'out_dim': output_dim,
                    'feat_dim': feat_dim,
                    'in_channels': in_dim,
                    'max_points': num_points,
                    'modality': 'bev'} 

        backbone = resnet.__dict__['resnet50'](param)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.head = MAC(outdim=output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x['out']
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.head(x)
        return x
  
    def __str__(self):
        return "ResNet50MAC"
    


class SPVMAC(nn.Module):
    def __init__(self, output_dim=256,feat_dim = 16,**argv):
        super(SPVMAC, self).__init__()

        self.backbone = spvcnn(output_dim=feat_dim)
        self.head = MAC(outdim=output_dim)

    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.backbone(x)
        y = torch.split(x, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 2, 0)
        x = self.head(x)
        return x #, y[:2]

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.head.parameters()
  
    def __str__(self):
        return "SPVMAC"