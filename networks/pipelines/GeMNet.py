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
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.head(x.permute(0,2,1))
        return x

  
    def __str__(self):
        return "ResNet50GeM"
    

class SPVGeM(nn.Module):
    def __init__(self, output_dim=256,feat_dim = 16,**argv):
        super(SPVGeM, self).__init__()

        self.backbone = spvcnn(output_dim=feat_dim)
        self.head = GeM(outdim=output_dim, p=3, eps=1e-6)

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
        return "SPVGeM"
