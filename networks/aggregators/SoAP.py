"""

Semantic Segmentation with Second-Order Pooling 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _l2norm(x):
    x = nn.functional.normalize(x, p=2, dim=-1)
    return x
    
class SoAP(nn.Module):
    def __init__(self, 
                 epsilon=1e-12, 
                 do_fc  = True, 
                 do_log = True, 
                 do_pn  = True,
                 do_pnl = True,
                 do_epn = True,
                 input_dim  = 16, 
                 output_dim = 256,
                 pn_value   = 0.75,
                 **kwargs):
        super(SoAP, self).__init__()
        
        self.do_fc = do_fc
        self.do_log = do_log
        self.do_epn = do_epn
        self.do_pn = do_pn
        self.do_pnl = do_pnl
        # power norm over  eigen-value power normalization
        self.do_epn = False if do_pn else self.do_epn
        
        self.input_dim = input_dim
        self.epsilon = epsilon
        self.fc = nn.LazyLinear( output_dim)
        if do_pnl:
            self.p = nn.Parameter(torch.ones(1) * pn_value)
        else:
            self.p = pn_value
        
    
    def __str__(self):
        
        pn_str = "pn:{:.2f}".format(self.p)  if not self.do_pnl else "pnl" 
        if self.do_epn and not self.do_pn:
            pn_str = "epn"
        elif not self.do_epn and (not self.do_pn and not self.do_pnl):
            pn_str = "no_pn"
        
        stack = ["SoAP" ,
                 "log" if self.do_log else "no_log",
                 pn_str,
                  "fc" if self.do_fc else "no_fc",
                 ]
        return '-'.join(stack)
    
    def _epn(self,x):
        """
        Eigen-value Power Normalization over the positive semi-definite matrix.
        """
        x = x.double()
        u_, s_, v_ = torch.svd(x)
        s_alpha = torch.pow(s_, 0.5)
        x =torch.matmul(torch.matmul(u_, torch.diag_embed(s_alpha)), v_.transpose(2, 1))
        return x#.float()
  
            
    def _log(self,x):
        # Log-Euclidean Tangent Space Mapping
        # Inspired by -> Semantic Segmentation with Second-Order Pooling
        # Implementation -> https://stackoverflow.com/questions/73288332/is-there-a-way-to-compute-the-matrix-logarithm-of-a-pytorch-tensor
        # x must be a symmetric positive definite (SPD) matrix
        x = x.double()
        #x = x + torchself.epsilon
        u, s, v = torch.linalg.svd(x)
        x=torch.matmul(torch.matmul(u, torch.diag_embed(torch.log(s))), v)
            
        return x

    def _pow_norm(self,x):
        # Power Normalization.
        if self.do_pnl:
            self.p.clamp(min=self.epsilon, max=1.0)
            
        x = torch.sign(x)*torch.pow(torch.abs(x),self.p)
            
        return x
    
    
    def forward(self, x):
            
        # Outer product
        batchSize, nPoints, dimFeat = x.data.shape
        x = x.unsqueeze(-1)
        x = x.matmul(x.transpose(3, 2))

        # Averaging over the points
        x = torch.mean(x, 1) 

        if self.do_log:
            x = self._log(x)
            
        if self.do_pn or self.do_pnl:
            x = self._pow_norm(x)
        
        if self.do_epn:
            x = self._epn(x)
        
        # Flatten
        x = x.reshape(batchSize, -1)   
        x = x.float()
        
        if self.do_fc:
            assert x.shape[1] == 256
            x =  self.fc(x)
        
        x = _l2norm(x)
        return torch.squeeze(x).float()
    
   
 