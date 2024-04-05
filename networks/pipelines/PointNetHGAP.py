import os
import sys
import torch
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


class MSGAP(nn.Module):
    def __init__(self, **argv):
        super(MSGAP, self).__init__()
        
        # Default stages
        self.stage_1 = argv['stage_1']
        self.stage_2 = argv['stage_2']
        self.stage_3 = argv['stage_3']
        
        self.head1 = GAP(outdim=argv['output_dim'])
        self.head2 = GAP(outdim=argv['output_dim'])
        self.head3 = GAP(outdim=argv['output_dim'])
        
        self.fco = nn.LazyLinear(argv['output_dim'])
        
        
    
    def forward(self, xi,xh,xo):
        d = torch.tensor([],dtype=xi.dtype,device=xi.device)
        if self.stage_1:
            xi = self.head1(xi)
            d = torch.cat((d, xi), dim=1)
   
        
        if self.stage_2:
            xh = self.head2(xh)
            d = torch.cat((d, xh), dim=1)
   
        if self.stage_3:
            xo = self.head3(xo)
            d = torch.cat((d, xo), dim=1)
        
        # L2 normalize
        d = self.fco(d)
        d = d / (torch.norm(d, p=2, dim=1, keepdim=True) + 1e-10)
        return d
    
    def __str__(self):
        return "MSGAP_S{}{}{}".format(int(self.stage_1),int(self.stage_2),int(self.stage_3))
    


class PointNetHGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=1024, **argv):
        super(PointNetHGAP, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        
        self.head= MSGAP(output_dim=output_dim, **argv)

    def forward(self, x):
        xo = self.point_net(x)
        
        h = self.point_net.t_out_h1
        h = h.transpose(1, 2)
        
        d = self.head(x,h,xo)
        return d
  
    def __str__(self):
        return "PointNetHGAP_{}_{}_{}".format(self.feat_dim, self.output_dim,
                                                   self.head.__str__())

