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


class SoGA(nn.Module):
    def __init__(self, **argv):
        super(SoGA, self).__init__()
        
        # Default stages
        self.fout = nn.LazyLinear(argv['outdim'])
        self.out  = None
        
    
    def forward(self, xi,xh,xo):
     

        xox = xo.transpose(1, 2)
        sxo = compute_similarity_matrix(xox)
        
        sxo = vectorized_euclidean_distance(sxo)
        sxo = sxo.flatten(1)

        d_out = self.fout(sxo)
        # L2 normalize
        x = nn.functional.normalize(d_out, p=2, dim=-1)    
        return x
    
    def __str__(self):
        return "SoGA_sym_dist"
    
    
    
    
class MSGAP(nn.Module):
    def __init__(self, **argv):
        super(MSGAP, self).__init__()
        
        # Default stages
        self.stage_1 = False #argv['stage_1']
        self.stage_2 = True #argv['stage_2']
        self.stage_3 = True #argv['stage_3']
        
        self.head1 = GAP(outdim=argv['outdim'])
        self.head2 = GAP(outdim=argv['outdim'])
        self.head3 = GAP(outdim=argv['outdim'])
        
        self.f1 = nn.LazyLinear(argv['outdim'])
        self.fout = nn.LazyLinear(argv['outdim'])
        self.out  = None
        
    
    def forward(self, xi,xh,xo):
     
        # Head's inout shape: BxNxF
        d = torch.tensor([],dtype=xi.dtype,device=xi.device)
        
        if self.stage_2:
            xh = xh.transpose(1, 2)
            sxh = compute_similarity_matrix(xh)
            sxh = sxh.flatten(1)
            d = torch.cat((d, sxh), dim=1)
   
        if self.stage_2:
            xox = xo.transpose(1, 2)
            sxo = compute_similarity_matrix(xox)
            sxo = sxo.flatten(1)
            d = torch.cat((d, sxo), dim=1)
        
        # outer product
        d = self.fout(d)
        d = d.unsqueeze(-1)
        d_d = torch.matmul(d , d.transpose(1, 2))
        dd_nom  = torch.softmax(d_d, dim=1)

        
        # Descriptor
        d_out = torch.mean(xo,-1)
        d_out = self.f1(d_out).unsqueeze(-1)
        d = torch.matmul(d_out.transpose(2,1), dd_nom).squeeze()
        # L2 normalize
        self.out = d / (torch.norm(d, p=2, dim=1, keepdim=True) + 1e-10)
        
        return d
    
    def __str__(self):
        return "MSGAP_attention_S{}{}{}".format(int(self.stage_1),int(self.stage_2),int(self.stage_3))
    

def mlp(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
        
    return layers  
            
class segment_classifier(nn.Module):
    def __init__(self,n_classes=7, feat_dim=256,kernels = [256, 128], **argv):
        super().__init__()
        
        self.n_classes = n_classes
        assert feat_dim 
        self.mlp = torch.nn.Sequential(*mlp(feat_dim, kernels, b_shared=False, bn_momentum=0.01, dropout=0.0))
        self.out_fc = torch.nn.Linear(kernels[-1], n_classes)
   
    
    def forward(self, descriptor, **argv):
        x = self.mlp(descriptor)
        out = self.out_fc(x)
        return out
    def __str__(self):
        return "segment_loss"

class PointNetHGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=256, **argv):
        super(PointNetHGAP, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.backbone = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.head = SoGA(outdim=output_dim)
        self.classifier = segment_classifier(n_classes=argv['n_classes'],feat_dim=output_dim,kernels=[256,64])
   
        

    def forward(self, x):
        # In Point cloud shape: BxNx3
        xo = self.backbone(x)
        xh = self.backbone.t_out_h1
        d = self.head(x,xh,xo)
        c = self.classifier(d)
        #d = self.head(xo)
        return d,c
  
    def __str__(self):
        return "PointNetHGAP_{}_{}_{}".format(self.feat_dim, self.output_dim,
                                                   self.head.__str__())

