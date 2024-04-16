import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
#from torch_geometric.nn import PointConv

import torch
from torch.nn import Linear
import torch.nn.functional as F

"""
https://colab.research.google.com/drive/1oO-Raqge8oGXGNkZQOYTH-je4Xi1SFVI?usp=sharing#scrollTo=i3G7KOpFIhjZ

"""

def _l2norm(x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

def flatten(x):
    return x.view(x.size(0), -1)

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
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

class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out

class PointNet_features(torch.nn.Module):
    def __init__(self,in_dim=3, dim_k=1024, scale=1):
        super().__init__()
        mlp_h1 = [int(32/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale)]
        mlp_h3 = [int(128 + 64), int(256),dim_k]
        
        self.h1 = MLPNet(in_dim, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.h3 = MLPNet(mlp_h3[0], mlp_h3, b_shared=True).layers

        self.flatten = Flatten()
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """ points -> features
            [B, N, 3] -> [B, K]
        """

        x = points.transpose(1, 2) # [B, 3, N]
        x2 = self.h1(x)
        self.t_out_h1 = x2 # local features
        x3 = self.h2(x2)
        xx = torch.cat([x2,x3],dim=1)
        x4 = self.h3(xx)
        xx = torch.cat([xx,x4],dim=1)
        return xx



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
    squared_euclidean_dist[squared_euclidean_dist < 0] = 0
    # Take the square root to get the Euclidean distances
    euclidean_dist = torch.sqrt(squared_euclidean_dist + 1e-10)
    return euclidean_dist


class PointNetKeypoint(nn.Module):
    def __init__(self):
        super(PointNetKeypoint, self).__init__()
        
        # Define a PointConv layer for feature extraction
        self.backbone = PointNet_features(in_dim=3, dim_k=512, scale=1)
        # Define fully connected layers for keypoint prediction
        self.fc1 = nn.LazyLinear(512)
        self.fc2 = nn.LazyLinear(256)  # Output keypoint coordinates
        self.d = None
    
    def __str__(self):
        return "KeypointExtractor_S011"
    
    
    def forward(self, x):
        # Apply PointConv layers for feature extraction
        #x = x.permute(0, 2, 1)
        x = self.backbone(x)
        
        self.d = torch.mean(x,dim=2)
        
        x1 = torch.relu(self.fc1(self.d))
        x1 = self.fc2(x1)
        
        return x1



class PointNetKeypointNET(nn.Module):
    def __init__(self,in_dim=3, dim_k=1024, scale=1):
        super(PointNetKeypointNET, self).__init__()
        
        self.mlp_h1 = [int(64/scale),int(128/scale),int(512/scale)]
        self.mlp_h2 = [int(512/scale),dim_k]
        #mlp_h3 = [int(128 + 64), int(256),dim_k]
        
        self.h1 = MLPNet(in_dim, self.mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(self.mlp_h1[-1], self.mlp_h2, b_shared=True).layers

        self.fc1 = nn.LazyLinear(256)
        self.d = None
    
    def __str__(self):
        return f"PointNetKeypointNET_{self.mlp_h1[-1]}_{self.mlp_h2[-1]}"
    
    
    def forward(self, x):
        # Apply PointConv layers for feature extraction
        #x = x.permute(0, 2, 1)
        x = x.transpose(1, 2) # [B, 3, N]
        x2 = self.h1(x)
        self.t_out_h1 = x2 # local features
        x3 = self.h2(x2)
        
        att = torch.matmul(x3,x2.transpose(2,1))
        x_norm = torch.softmax(att,dim=2)
        
        self.d = torch.mean(x3,dim=2)
        
        self.d = torch.matmul(torch.unsqueeze_copy(self.d,1),x_norm)
        
        x1 = _l2norm(self.fc1(self.d).squeeze())
        
        return x1
        
    