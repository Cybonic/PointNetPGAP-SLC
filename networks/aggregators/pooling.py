#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
#from ..utils import *

'''
Code taken from  https://github.com/slothfulxtx/TransLoc3D 
'''

def _l2norm(x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

class MAC(nn.Module):
    def __init__(self,outdim=256, **rgv):
        super().__init__()
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        x = x.view(x.shape[0],x.shape[1],-1)
        x = torch.max(x, dim=-1, keepdim=False)[0]
        return _l2norm(self.fc(x))


class GeM(nn.Module):
    def __init__(self, outdim=256, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        #self.p = p
        self.eps = eps
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # This implicitly applies ReLU on x (clamps negative values)
        x = x.clamp(min=self.eps).pow(self.p)
        
        x = x.view(x.shape[0],x.shape[1],-1)
        x = F.avg_pool1d(x, x.size(-1))
       
        x = x.view(x.shape[0],x.shape[1])
        
        x = torch.pow(x,1./self.p)
        x = _l2norm(self.fc(x))
        return x # Return (batch_size, n_features) tensor











