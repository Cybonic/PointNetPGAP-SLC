
'''
https://arxiv.org/pdf/1801.06761.pdf -> PU-Net: Point Cloud Upsampling Network

https://github.com/yulequan/PU-Net

https://github.com/IAmSuyogJadhav/PointNormalNet/blob/main/main/README.md



'''


from .normal_estimation.code.model import PointCloudNet,PointnetSAModuleMSG
import torch.nn as nn
import torch
from  ..aggregators import GAP
from ..aggregators import SoAP


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
    #X = X.transpose(2, 1)
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

def _l2norm(x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x
    
class PointNormalNet(nn.Module):
    def __init__(self,input_channels = 3, num_points = 10000, use_xyz = False, **argv):
        super(PointNormalNet, self).__init__()
        
        
        num_points = int(num_points)
        
        from .normal_estimation.code.model import PointCloudNet
        self.model = PointCloudNet(input_channels = 0, output_channels = 6, num_points = num_points)

        self.fc_out = nn.LazyLinear(256)
        #self.head  =  SoAP.SoAP(input_dim = 20, 
        #                   output_dim = 256, 
        #                   do_fc = True, 
        #                   do_log = False, 
        #                   do_pn = False, 
        #                   do_pnl = False, 
        #                   do_epn = True)
        
    
        
    def get_pipeline(self):
        return self.pipeline
    
    def forward(self, x):
        
        l = x.contiguous()
        d = self.model(l)
        #gd = s.pop('global')
        
        #out = s['1'].contiguous()
        
        #d = torch.mean(s, dim = -1)
        #s = compute_similarity_matrix(out)
        #d = s
        #d = vectorized_euclidean_distance(s)
        #cov = torch.matmul(out.transpose(1,2), out)

        #s = s.transpose(1,2).contiguous()
        #s = s[:,:,:20]
        #print(s.shape)
        #d = d.view(d.shape[0],-1)
        d = _l2norm(self.fc_out(d))
        #d = _l2norm(d)
        return d
       
    
    def __str__(self):
        return f"PointNormalNet_PointNet_1204_256_RBF_1024_16"