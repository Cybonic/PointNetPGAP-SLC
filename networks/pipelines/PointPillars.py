
from .pointpillars.model.pointpillars import PointPillars 
CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
import torch
import torch.nn as nn
import os

class PointPillarsGAP(nn.Module):
    def __init__(self, **argv):
        super().__init__()
        
        # get current path
        path = os.path.dirname(os.path.realpath(__file__))
        chkpt = os.path.join(path,'pointpillars/model/epoch_160.pth')
        self.model = PointPillars(nclasses=len(CLASSES)).cuda()
        self.model.load_state_dict(torch.load(chkpt))
        
        self.fc2 = nn.LazyLinear(256)  # Output keypoint coordinates
        
        # freeze the model parameters except backbone
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.neck.parameters():
            param.requires_grad = True
        
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    def __str__(self):
        return "PointPillarsGAP"
    
    def forward(self, x):
        # add additional reflection dim with zeros to the input cloud
        b,n,d = x.shape
        
        x = torch.cat([x, torch.zeros((b,n,1),device=x.device)], dim=2,)
        
        f = self.model(x)
        f = torch.mean(f, dim=1).flatten(1)
        d = self.fc2(f)
        # normalize the output
        d = d / torch.norm(d, dim=1, keepdim=True)
        
        return d