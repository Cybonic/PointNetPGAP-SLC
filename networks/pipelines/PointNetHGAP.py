import os
import sys
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.GAP import *
from ..backbones.pointnet import *
from networks.utils import *


def so_meanpool(x):

    batchSize, nFeat, dimFeat = x.data.shape
    #x = torch.reshape(x, (-1, dimFeat))
    x = torch.unsqueeze(x, -1)
    x = x.matmul(x.transpose(3, 2))
    #x = torch.reshape(x, (batchSize, nFeat, dimFeat, dimFeat))
    x = torch.mean(x, 1)
    return x
        
class PointNetHGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=1024, **argv):
        super(PointNetHGAP, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=8)
        
        
        self.head = GAP(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x)
        # Transpose to [B, N, C]
        x = x.transpose_(1, 2)
        h = self.point_net.t_out_h1
        xx = so_meanpool(x)
        x = self.head(x)
        
        
        return x
  
    def __str__(self):
        return "PointNetHGAP_{}_{}".format(self.feat_dim, self.output_dim)

