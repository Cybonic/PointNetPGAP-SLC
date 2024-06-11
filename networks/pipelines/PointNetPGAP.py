
'''
https://arxiv.org/pdf/1801.06761.pdf -> PU-Net: Point Cloud Upsampling Network

https://github.com/yulequan/PU-Net

https://github.com/IAmSuyogJadhav/PointNormalNet/blob/main/main/README.md



'''

import torch.nn as nn
import torch
from backbones.pointnet import PointNet_features

def _l2norm(x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x
    
class PointNetPGAP(nn.Module):
    def __init__(self,input_channels = 3, num_points = 10000, use_xyz = False, **argv):
        super(PointNetPGAP, self).__init__()
        
        self.feat =  argv['output_channels']
        
        self.GLOBAL_module  = PointNet_features(in_dim = input_channels, dim_k = argv['output_channels'], use_tnet = False, scale = 1)
        self.num_points = int(num_points)

        self.fc_out = nn.LazyLinear(256)

        
    def get_pipeline(self):
        return self.pipeline
    
    def forward(self, x):
        
        x = self.GLOBAL_module(x)
        xmean = torch.mean(x, dim=-1)
        
        x = x.matmul(x.transpose(2, 1))/(self.num_points-1)

        x_cov = x.flatten(start_dim=1)
        
        rfb = torch.cat([xmean.squeeze(),x_cov], dim=-1)
        
        d = _l2norm(self.fc_out(rfb))
        return d
       
    
    def __str__(self):
        return f"PointNetPGAP"