import os
import sys
import torch
import torch.nn as nn
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from networks.aggregators.SoAP import *
from ..backbones.pointnet import *
from networks.backbones.spvnas.model_zoo import spvcnn
from networks.pipelines.pipeline_utils import *

__all__ = ['SPVSoAP3D']  #if a client imports this module using from SPVSoAP3D import *, only the SPVSoAP3D attribute or function will be imported

    
class SPVSoAP3D(nn.Module):
    def __init__(self, output_dim=256,
                 local_feat_dim=16,
                 pres=1,
                 vres=1,
                 cr=0.64,
                 **kwargs):
        super(SPVSoAP3D, self).__init__()

        self.backbone = spvcnn(output_dim=local_feat_dim,pres=pres,vres=vres,cr=cr)
        
        self.head = SoAP(
                        input_dim=local_feat_dim,
                        output_dim=output_dim,
                        **kwargs)
        
        
    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)
        x = self.backbone(x)
        y = torch.split(x, list(counts))
        lfeat = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 0, 2)
        x = self.head(lfeat)
        
        #out = {'out':x,'feat':lfeat}
        return x #, y[:2]

    def __str__(self):
        
        stack = ["SPVSoAP3D" ,
                 str(self.head),
                 ]
        return '-'.join(stack)
    
    


class PointNetSoAP3D(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024,**argv):
        super(PointNetSoAP3D, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        self.head = SoAP(
                        input_dim=feat_dim,
                        output_dim=output_dim,
                        **argv)

    def forward(self, x):
        x = self.point_net(x)
        x = self.head(x)
        return x
    
    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.head.parameters()
  
    def __str__(self):
        return "PointNetSoAP3D"

