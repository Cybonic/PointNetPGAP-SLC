
'''
https://arxiv.org/pdf/1801.06761.pdf -> PU-Net: Point Cloud Upsampling Network

https://github.com/yulequan/PU-Net

https://github.com/IAmSuyogJadhav/PointNormalNet/blob/main/main/README.md



'''


from .normal_estimation.code.model import PointCloudNet,PointnetSAModuleMSG
import torch.nn as nn



class PointNormalNet(nn.Module):
    def __init__(self,input_channels = 3, num_points = 10000, use_xyz = True, **argv):
        super(PointNormalNet, self).__init__()
        
        
        num_points = int(num_points)
        
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.1,],
                nsamples=[32,],
                mlps=[[c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        
        c_out_0 = 64
        num_points = int(num_points / 4)
        c_in = c_out_0
        #self.SA_modules.append(
        #    PointnetSAModuleMSG(
        #        npoint=num_points,
        #        radii=[0.2,],
        #        nsamples=[32,],
        #        mlps=[[c_in, 64, 64, 128]],
        #        use_xyz=use_xyz,
        #    )
        #)
        
        
    def get_pipeline(self):
        return self.pipeline
    
    def forward(self, x):
        
        self.SA_modules[0](x)
        
    
        return self.pipeline(x)
    
    def __str__(self):
        return "PointNormalNet"