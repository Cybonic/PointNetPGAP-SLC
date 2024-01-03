import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.pooling import *
from ..backbones.pointnet import *
from ..backbones import resnet
from networks.utils import *
from networks.aggregators import SOP

def _l2norm(x):
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x
    
def so_cov(x):
    batchSize, nFeat, dimFeat = x.data.shape
    x = torch.reshape(x, (-1, dimFeat))
    x = torch.unsqueeze(x, -1)
    mean = torch.mean(x, 0,keepdim=True)
    x = x-mean
    
    x = x.matmul(x.transpose(1, 2))

    x = torch.reshape(x, (batchSize, nFeat, dimFeat, dimFeat))
    x = torch.mean(x, 1)
    x = torch.reshape(x, (-1, dimFeat, dimFeat))
    
    x = x.double()

    # For pytorch versions < 1.9
    u_, s_, v_ = torch.svd(x)
    s_alpha = torch.pow(s_, 0.5)
    x = u_ @ torch.diag_embed(s_alpha) @ v_.transpose(-2, -1)

    x = torch.reshape(x, (batchSize, dimFeat * dimFeat))
    return x 
    
class PointNetSOP(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=10000, use_tnet=False, output_dim=256):
        super(PointNetSOP, self).__init__()

        self.feat_dim = feat_dim
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        self.head = so_cov
        self.fc =  nn.LazyLinear(output_dim)
    
    
    
    def forward(self, x):
        x = self.point_net(x)
        x = x.permute(0,2,1)
        x = self.head(x)
        x = self.fc(x.float())
        return _l2norm(x)
    
    def __str__(self):
        return f"PointNetSOP-FDim{self.feat_dim}"


class ResNet50MAC(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, output_dim=1024):
        super(ResNet50MAC, self).__init__()

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
        
        self.head = MAC(outdim=output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x['out']
        x = self.head(x)
        return x
    
    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.net_vlad.parameters()
  
    def __str__(self):
        return "ResNet50MAC"
    

