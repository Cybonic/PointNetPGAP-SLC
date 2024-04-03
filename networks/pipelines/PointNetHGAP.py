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
    S = vectorized_euclidean_distance(S)
    return S.view(S.shape[0],-1)


def compute_similarity_matrix(X):
    batch,nfeat,feat_dim = X.shape
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
        
class PointNetHGAP(nn.Module):
    def __init__(self, feat_dim = 1024, use_tnet=False, output_dim=1024, **argv):
        super(PointNetHGAP, self).__init__()

        self.feat_dim = feat_dim
        self.output_dim = output_dim
        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=8)
        
        
        self.fco = nn.LazyLinear(output_dim)
        self.fch = nn.LazyLinear(output_dim)
        self.fci = nn.LazyLinear(output_dim)
        
        # Default stages
        self.stage_1 = False
        self.stage_2 = False
        self.stage_3 = True
        
        if 'stage_1' in argv:
            self.stage_1 = argv['stage_1']
        if 'stage_2' in argv:
            self.stage_2 = argv['stage_2']
        if 'stage_3' in argv:
            self.stage_3 = argv['stage_3']
        
        self.head = GAP(outdim=output_dim)

    def forward(self, x):
        xo = self.point_net(x)
        
        h = self.point_net.t_out_h1
        h = h.transpose(1, 2)
        
        d = torch.zeros(h.shape[0], 256, device=h.device)
        if self.stage_1:
            #xi = x.transpose(1, 2)
            xi =wishart_descriptor(x)
            xi = self.fci(xi)
            xi = xi / (torch.norm(xi, p=2, dim=1, keepdim=True) + 1e-10)
            d += xi
        
        if self.stage_2:
            xh =wishart_descriptor(h)
            xh = self.fch(xh)
            xh = xh / (torch.norm(xh, p=2, dim=1, keepdim=True) + 1e-10)
            d += xh
        
        # Transpose to [B, N, C]
        if self.stage_3:
            xo = xo.transpose(1, 2)
            xo =wishart_descriptor(xo)
            xo = self.fco(xo)
            xo = xo / (torch.norm(xo, p=2, dim=1, keepdim=True) + 1e-10)
            d += xo
        
        # L2 normalize
        d = d / (torch.norm(d, p=2, dim=1, keepdim=True) + 1e-10)
        return d
  
    def __str__(self):
        return "PointNetHGAP_{}_{}_S{}{}{}".format(self.feat_dim, self.output_dim,
                                                   int(self.stage_1),int(self.stage_2),int(self.stage_3))

