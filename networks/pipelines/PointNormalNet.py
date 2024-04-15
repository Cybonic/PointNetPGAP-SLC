from .normal_estimation.code.model import PointCloudNet
import torch.nn as nn



class PointNormalNet(nn.Module):
    def __init__(self,**argv):
        super(PointNormalNet, self).__init__()
        self.pipeline = PointCloudNet(**argv)
        
    def get_pipeline(self):
        return self.pipeline
    
    def forward(self, x):
        return self.pipeline(x)
    
    def __str__(self):
        return "PointNormalNet"