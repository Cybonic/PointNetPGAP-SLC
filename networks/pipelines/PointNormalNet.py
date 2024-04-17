
'''
https://arxiv.org/pdf/1801.06761.pdf -> PU-Net: Point Cloud Upsampling Network

https://github.com/yulequan/PU-Net

https://github.com/IAmSuyogJadhav/PointNormalNet/blob/main/main/README.md



'''


from .normal_estimation.code.model import PointCloudNet,PointnetSAModuleMSG
import torch.nn as nn
from  ..aggregators import GAP
from ..aggregators import SoAP



class PointNormalNet(nn.Module):
    def __init__(self,input_channels = 3, num_points = 10000, use_xyz = False, **argv):
        super(PointNormalNet, self).__init__()
        
        
        num_points = int(num_points)
        
        from .normal_estimation.code.model import PointCloudNet
        self.model = PointCloudNet(input_channels = 0, output_channels = 6, num_points = num_points)
        self.head  =  GAP.GAP()
        #self.head  =  SoAP.SoAP(input_dim = 256, 
        #                   output_dim = 256, 
        #                   do_fc = False, 
        #                   do_log = True, 
        #                   do_pn = False, 
        #                   do_pnl = False, 
        #                   do_epn = False)
        
    
        
    def get_pipeline(self):
        return self.pipeline
    
    def forward(self, x):
        
        l = x.contiguous()
        s = self.model(l)
        
        return self.head(s)
       
    
    def __str__(self):
        return f"PointNormalNet_MAX_{self.head}"