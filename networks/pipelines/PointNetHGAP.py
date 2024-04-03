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
        self.fc = nn.LazyLinear(output_dim)
        
        self.head = GAP(outdim=output_dim)

    def forward(self, x):
        x = self.point_net(x)
        # Transpose to [B, N, C]
        x = x.transpose(1, 2)
        x =wishart_descriptor(x)
        x = self.fc(x)
        # L2 normalize
        x = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-10)
        return x
  
    def __str__(self):
        return "PointNetHGAP_{}_{}".format(self.feat_dim, self.output_dim)

