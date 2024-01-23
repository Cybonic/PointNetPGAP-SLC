import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.GAP import *
from ..backbones.pointnet import *
from networks.utils import *

class PointNetGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=1024, **argv):
        super(PointNetGAP, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.head = GAP(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x)
        x = self.head(x)
        return x
  
    def __str__(self):
        return "PointNetGAP"

