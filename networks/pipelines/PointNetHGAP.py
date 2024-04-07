import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ..aggregators.GAP import *
from ..backbones.pointnet import *
from networks.utils import *

def wishart_descriptor(X):
    
    n_samples = X.shape[0]
    dim = X.shape[1]
    #X = X.t()
    S = compute_similarity_matrix(X)
    #S = vectorized_euclidean_distance(S)
    return S.view(S.shape[0],-1)


def compute_similarity_matrix(X):
    batch,nfeat,feat_dim = X.shape
    X = X.to(torch.float16)
    #print(n_samples)
    if not torch.is_tensor(X):
        X = torch.tensor(X, dtype=torch.float32)
    # Compute similarity matrix
    #S = X@X.t()
    X = X.transpose(2, 1)
    S = X.matmul(X.transpose(2, 1))
    return S/nfeat


def vectorized_euclidean_distance(S):
    # Convert to PyTorch tensor if not already
    
    if not torch.is_tensor(S):
        S = torch.tensor(S, dtype=torch.float32)
    # Compute the squared Euclidean distances
    #squared_euclidean_dist = torch.diag(S)[:, None] - 2 * S + torch.diag(S)[None, :]
    batch_eye = torch.eye(S.shape[1], device=S.device).repeat(S.shape[0], 1, 1)
    diag = torch.diag_embed(torch.diagonal(S, dim1=-2, dim2=-1))
    squared_euclidean_dist = diag - 2*S + diag.transpose(2,1)
    
    # Ensure non-negative distances due to numerical precision issues
    #squared_euclidean_dist[squared_euclidean_dist < 0] = 0
    # Take the square root to get the Euclidean distances
    #euclidean_dist = torch.sqrt(squared_euclidean_dist + 1e-10)
    return squared_euclidean_dist

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
        self.stage_1 = False #argv['stage_1']
        self.stage_2 = False #argv['stage_2']
        self.stage_3 = False #argv['stage_3']
        
        self.head1 = GAP(outdim=argv['output_dim'])
        self.head2 = GAP(outdim=argv['output_dim'])
        self.head3 = GAP(outdim=argv['output_dim'])
        
        self.f1 = nn.LazyLinear(512)
        self.fout = nn.LazyLinear(argv['output_dim'])
        self.out  = None
        
    
    def forward(self, xi,xh,xo):
     
        # Head's inout shape: BxNxF
        d = torch.tensor([],dtype=xi.dtype,device=xi.device)
        
        if self.stage_1:
            sxi=compute_similarity_matrix(xi).unsqueeze(1)
            xi = torch.mean(xi,-1)
            d = torch.cat((d, sxi), dim=1)
   
        
        if self.stage_2:
            xh = xh.transpose(1, 2)
            sxh = so_meanpool(xh).unsqueeze(1)
            sxh = vectorized_euclidean_distance(sxh).unsqueeze(1)
            #sxh=compute_similarity_matrix(xh).unsqueeze(1)
            d = torch.cat((d, sxh), dim=1)
   
        if self.stage_2:
            xox = xo.transpose(1, 2)
            sxo = so_meanpool(xox)
            #sxo = vectorized_euclidean_distance(sxo).unsqueeze(1)
            self.fco(sxo)
            #xo = torch.mean(xo,-1)
            d = torch.cat((d, sxo), dim=1)
        xo=torch.mean(xo,-1) 
        # L2 normalize
        out = self.f1(xo)
        self.out = out / (torch.norm(out, p=2, dim=1, keepdim=True) + 1e-10)
        d = self.fout(torch.relu(out))
        return d
    
    def __str__(self):
        return "MSGAP_2stage_out_S{}{}{}".format(int(self.stage_1),int(self.stage_2),int(self.stage_3))
    


class PointNetHGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=256, **argv):
        super(PointNetHGAP, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        
        self.head= MSGAP(output_dim=output_dim, **argv)
        self.out = None
        #self.head = GAP(outdim=output_dim)

    def forward(self, x):
        # In Point cloud shape: BxNx3
        xo = self.point_net(x)
        
        # backbone output shape: BxFxN
        #xo = xo.transpose(1, 2)
        
        h = self.point_net.t_out_h1
        x = x.transpose(1, 2)
        
        # Head's Input shape: BxFxN
        d = self.head(x,h,xo)
        self.out = self.head.out
        #d = self.head(xo)
        return d
  
    def __str__(self):
        return "PointNetHGAP_{}_{}_{}".format(self.feat_dim, self.output_dim,
                                                   self.head.__str__())

