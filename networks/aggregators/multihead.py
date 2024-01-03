#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .NetVLAD import NetVLADLoupe
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


class SPoC(nn.Module):
    def __init__(self, outdim=256,**argv):
        super().__init__()
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        # Swap the axis
        #x = x.permute(0, 2, 1)
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.fc(torch.mean(x, dim=-1, keepdim=False)) # Return (batch_size, n_features) tensor
        return _l2norm(x)


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

# MultiHead Aggregation
class MultiHead(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MultiHead,self).__init__()
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    
    self.fusion= nn.Parameter(torch.zeros(1,3))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    # print(self.fusion.data)
  def forward(self,x):
    spoc =  self.spoc(x)
    gem  =  self.gem(x)
    mac  =  self.mac(x)
    z    =  torch.stack([spoc,gem,mac],dim=1)
    fu = torch.matmul(self.fusion,z).squeeze()
    
    return _l2norm(fu)

class MultiHeadSGMaxPoolingFC(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MultiHeadSGMaxPoolingFC,self).__init__()
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    self.fc = nn.LazyLinear(outdim)
    #self.fusion= nn.Parameter(torch.zeros(1,3))
    # Initialization
    #nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    # print(self.fusion.data)
  def forward(self,x):
    spoc =  self.spoc(x)
    gem  =  self.gem(x)
    z    =  torch.stack([spoc,gem],dim=1)

    fu = torch.max(z,dim=1)[0]
    
    return _l2norm(self.fc(fu))

class VLADSPoCMaxPooling(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(VLADSPoCMaxPooling,self).__init__()
    self.spoc = SPoC(outdim=outdim)
    self.vlad  = NetVLADLoupe(feature_size=1024, max_samples=10000, cluster_size=64,
                                     output_dim=outdim, gating=True, add_batch_norm=True,
                                     is_training=True)
  def forward(self,x):
    spoc =  _l2norm(self.spoc(x))
    vlad  = _l2norm(self.vlad(x))
    z     =  torch.stack([spoc,vlad],dim=1)
    fu    = torch.max(z,dim=1)[0]
    return _l2norm(fu)

class VLADSPoCGeMMaxPooling(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(VLADSPoCMaxPooling,self).__init__()
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.vlad = NetVLADLoupe(feature_size=1024, max_samples=10000, cluster_size=64,
                                     output_dim=outdim, gating=True, add_batch_norm=True,
                                     is_training=True)
    
    self.fusion= nn.Parameter(torch.zeros(1,3))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    # print(self.fusion.data)

  def forward(self,x):
    spoc  =  _l2norm(self.spoc(x))
    vlad  = _l2norm(self.vlad(x))
    gem   = _l2norm(self.gem(x))
    z     =  torch.stack([spoc,gem,vlad],dim=1)
    fu = torch.max(z,dim=1)[0]
    return _l2norm(fu)
  
class VLADSPoCLearned(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(VLADSPoCLearned,self).__init__()
   
    self.vlad  = NetVLADLoupe(feature_size=1024, max_samples=10000, cluster_size=64,
                                     output_dim=outdim, gating=True, add_batch_norm=True,
                                     is_training=True)
    self.spoc  = SPoC(outdim=outdim)
    
    self.fusion= nn.Parameter(torch.zeros(1,2))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)
    
  def forward(self,x):
    spoc  = self.spoc(x)
    vlad  =  self.vlad(x)
    z    =  torch.stack([spoc,vlad],dim=1)
    fu   =  torch.matmul(self.fusion,z).squeeze()
    return _l2norm(fu)
  
  
class MultiHeadMAXPolling(nn.Module):
  def __init__(self,outdim=256,init_std=0.1):
    super(MultiHeadMAXPolling,self).__init__()
    self.outdim = outdim
    self.spoc = SPoC(outdim=outdim)
    self.gem  = GeM(outdim=outdim)
    self.mac  = MAC(outdim=outdim)
    
    self.fusion= nn.Parameter(torch.zeros(1,3))
    # Initialization
    nn.init.normal_(self.fusion.data, mean=0, std=init_std)

  def forward(self,x):
    spoc =  _l2norm(self.spoc(x))
    gem  =  _l2norm(self.gem(x))
    mac  =  _l2norm(self.mac(x))
    z    =  torch.stack([spoc,gem,mac],dim=1)
    fu = torch.max(z,dim=1)
    return _l2norm(fu[0])



if __name__=="__main__":
    
    head = MultiHeadMAXPolling(256)
    
    x = torch.randn(2,1024,10000)
    y = head(x)
    
    
    










