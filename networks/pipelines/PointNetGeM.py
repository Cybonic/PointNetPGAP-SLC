import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.NetVLAD import *
from ..backbones.pointnet import *

class PointNetGeM(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetGeM, self).__init__()
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.net_vlad = NetVLADLoupe(feature_size=feat_dim, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        x = self.point_net(x)
        x = self.net_vlad(x)
        return x
    
    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.net_vlad.parameters()
  
    def __str__(self):
        return "PointNetVLAD"
    

