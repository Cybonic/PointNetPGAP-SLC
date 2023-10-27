
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==========================================================================
#  KL divergence
# ==========================================================================
def pmf(input_tensor,tau = 1, eps=1e-8):
    log_probs = F.log_softmax(input_tensor/tau,dim=2)
    exp = torch.exp(log_probs).clone()
    exp[exp==0]=eps
    return(exp)

def logit_kl_divergence_loss(x, y, eps=1e-8, **argv):
    # Map to probabilistic mass function
    px = pmf(x)
    py = pmf(y)
    kl = px * torch.log2(px / py)
    return torch.max(torch.nansum(kl,dim=[2,1]),torch.tensor(eps))

def kl_divergence(p, q):
    kl = p * torch.log2(p / q)
    out = torch.nansum(kl,dim=[2,1])
    return out

# ==========================================================================
#  cosine
# ==========================================================================
def cosine_torch_loss(x,y,eps=1e-8,dim=0):
    #return torch.max(1-torch.abs(cosine(x,y,dim)),torch.tensor(eps))
    loss = 1 - F.cosine_similarity(x, y, dim, eps)
    return torch.max(loss,torch.tensor(eps))

def cosine_loss(x,y,eps=1e-8,dim=0):
    return torch.max(1-torch.abs(cosine(x,y,dim)),torch.tensor(eps))
    #value = 1-F.cosine_similarity(x, y, dim, eps)
    #return torch.max(value,torch.tensor(eps))
    
def cosine(a,b,dim=0):
    num = torch.tensordot(b,a,dims=([1,2],[1,2])).squeeze()
    den = (torch.norm(a,dim=2)*torch.norm(b,dim=2)).squeeze()
    return torch.div(num,den)

# ==========================================================================
#  Euclidean Distance
# ==========================================================================

def L2_np(a,b, dim=0):
    return np.sqrt(np.sum((a - b)**2,axis=dim))

def normal_kernal(value,sigma=0.9):
    x = 2*torch.pow(torch.tensor(sigma),2)
    return torch.exp(-(1/x)*value)
    
def L2_loss(a,b, dim=0, eps=1e-8):
    squared_diff = torch.pow((a - b),2)
    value = torch.sqrt(torch.sum(squared_diff,dim=dim)+eps)
    return torch.max(value,torch.tensor(eps))

def totensorformat(x,y):
    if not torch.is_tensor(x):
        x = torch.tensor(x,dtype=torch.float32)

    if len(x.shape)<3:
        bs = x.shape[0]
        x = x.view((bs,1,-1))

    if not torch.is_tensor(y):
        y = torch.tensor(y,dtype=torch.float32)

    if len(y.shape)<3:
        bs = y.shape[0]
        y = y.view(bs,1,-1)
    
    x = x.type(torch.float32)
    y = y.type(torch.float32)
    return(x,y)

    
def get_distance_function(name):
    if name == 'L2':              loss = L2_loss
    elif name == 'L2_np':         loss = L2_loss
    elif name == 'cosine':        loss = cosine_loss
    elif name == 'cosine_torch':  loss = cosine_torch_loss
    elif name == 'kl_divergence': loss = logit_kl_divergence_loss
    else:
        raise NameError
    return(loss)

#==================================================================================================
#
#
#==================================================================================================

class LazyTripletLoss():
  def __init__(self, metric= 'L2', margin=0.5 , eps=1e-8,**argv):

    self.margin = margin
    self.metric = metric
    self.eps = torch.tensor(eps)
    
    # Loss types
    # self.loss = L2_loss
    self.loss = get_distance_function(metric)
  def __str__(self):
    return type(self).__name__ + '_' + self.metric
 
  def __call__(self,descriptor = {},**args):
    
    #a_pose,p_pose,n_pose = pose[0],pose[1],pose[2]
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']

    assert a.shape[0] == 1
    assert p.shape[0] == 1, 'positives samples must be 1'

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)
    
    # Anchor - positive
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2).squeeze()
    # Anchor - negative
    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    #neg_loss_array =  normal_kernal(neg_loss_array)
    an = torch.min(neg_loss_array) # get the lowest negative distance (aka hard)
    s = ap - an + self.margin
    value = torch.max(self.eps,s)
    loss = value.clamp(min=0.0)

    return(loss,{'p':ap,'n':an})


class PositiveLoss():
  def __init__(self, metric= 'L2', margin=0.5 , eps=1e-8,**argv):

    self.margin = margin
    self.metric = metric
    self.eps = torch.tensor(eps)
    
    # Loss types
    # self.loss = L2_loss
    self.loss = get_distance_function(metric)
  def __str__(self):
    return type(self).__name__ + '_' + self.metric
 
  def __call__(self,descriptor = {},**args):
    
    #a_pose,p_pose,n_pose = pose[0],pose[1],pose[2]
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']

    assert a.shape[0] == 1
    assert p.shape[0] == 1, 'positives samples must be 1'

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)
    
    # Anchor - positive
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2).squeeze()


    s = ap #- an + self.margin
    value = torch.max(self.eps,s)
    loss = value.clamp(min=0.0)

    n = torch.ones_like(ap,dtype=torch.float32)*-1
    return(loss,{'p':ap,'n':n})


#==================================================================================================
#
#
#==================================================================================================

class LazyQuadrupletLoss():
  def __init__(self, metric= 'L2', margin1 = 0.5 ,margin2 = 0.5 , eps=1e-8, **argv):
    
    #assert isinstance(margin,list) 
    #assert len(margin) == 2,'margin has to have 2 elements'

    self.margin1 =  margin1 
    self.margin2 = margin2
    self.metric = metric
    self.eps = torch.tensor(eps)
    # Loss types
    #self.loss = L2_loss
    self.loss = get_distance_function(metric)
  
  def __str__(self):
    return type(self).__name__ + '_' + self.metric

  def __call__(self,descriptor = {},**args):
    
    #a_pose,p_pose,n_pose = pose[0],pose[1],pose[2]
    a,p,n = descriptor['a'],descriptor['p'],descriptor['n']
    assert a.shape[0] == 1
    #assert p.shape[0] == 1, 'positives samples must be 1'
    assert n.shape[0] >= 2,'negative samples must be at least 2' 

    if len(a.shape) < len(n.shape): 
        a = a.unsqueeze(dim=0)
    if len(p.shape) < len(n.shape): 
        p = p.unsqueeze(dim=0)
    if len(n.shape) < len(a.shape):
        n = n.unsqueeze(dim=0)

    # Anchor - positive
    a_torch,p_torch = totensorformat(a,p)
    ap = self.loss(a_torch, p_torch,dim=2)
    ap = normal_kernal(ap)
    # Anchor - negative
    a_torch,n_torch  = totensorformat(a,n)
    neg_loss_array = self.loss(a_torch,n_torch,dim=2)
    neg_loss_array = normal_kernal(neg_loss_array)
    # Hard negative
    n_hard_idx = [torch.argmin(neg_loss_array).cpu().numpy().tolist()] # get the negative with smallest distance (aka hard)
    an = neg_loss_array[n_hard_idx]
    
    # Random negative (NR)
    n_negs    = n_torch.shape[0]
    idx_arr   = np.arange(n_negs)
    elig_neg  = np.setxor1d(idx_arr,n_hard_idx) # Remove the hard negative index from the negative array  
    n_rand_idx  = torch.randint(0,elig_neg.shape[0],(1,)).numpy().tolist()
    dn_rand   = n_torch[n_rand_idx] # random negative descriptor
    
    nn_prime= self.loss(n_torch[elig_neg],dn_rand,dim=2) # among the negatives select subset of eligibles, and compute the distance between NR and all negatives  
    n_random_hard_idx = [torch.argmin(nn_prime).cpu().numpy().tolist()] # get the negative with smallest distance w.r.t NR (aka NRH)
    nr2h = nn_prime[n_random_hard_idx] # get the smallest distance between NR and NRH

    # Compute first term
    s1 = ap.squeeze() - an.squeeze() + self.margin1
    first_term = torch.max(self.eps,s1).clamp(min=0.0)
    # Compute second term
    s2 = ap.squeeze() - nr2h.squeeze() + self.margin2
    second_term = torch.max(self.eps,s2).clamp(min=0.0)
    # compute loss
    loss = first_term + second_term

    return(loss,{'t1':s1,'t2':s2,'n_p':nr2h})


