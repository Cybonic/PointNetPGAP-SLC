from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn

from ..torch_rbf import RBF, gaussian
from ..pointnet2.utils.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG

def compute_moments(pointcloud,dim=-1, eps=1e-8):
    # Assume pointcloud is of shape (num_points, 3)
    mean = torch.mean(pointcloud, dim=dim).unsqueeze(dim)
    dmean = (pointcloud - mean)
    # Compute the moments
    
    variance = torch.var(pointcloud, dim=dim).unsqueeze(dim)
    
    #aux = torch.pow(torch.sqrt(variance),3)
    #aux1 = torch.pow(variance,2)
    std = torch.sqrt(variance + eps)
    skewness = torch.mean(torch.pow(dmean,3), dim=dim).unsqueeze(dim) 
    skewness = skewness/(torch.pow(std,3))
    
    kurtosis = torch.mean(torch.pow(dmean,4), dim=dim).unsqueeze(dim) # / aux1 - 3
    kurtosis = kurtosis / (torch.pow(std,4))
    kurtosis = kurtosis - 3
    return torch.cat([mean.squeeze(), variance.squeeze(), skewness.squeeze(), kurtosis.squeeze()], dim=dim)



class GAP(nn.Module):
    def __init__(self, input = 1024,outdim=256,**argv):
        super().__init__()
        self.fc = nn.Linear(input,outdim)

    def __str__(self):
        return "GAP"
    
    def forward(self, x):
        # Return (batch_size, n_features) tensor
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.fc(torch.mean(x, dim=-1, keepdim=False)) # Return (batch_size, n_features) tensor
        return nn.functional.normalize(x, p=2, dim=-1)


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()
   
    def forward(self, x):
        return x.view(x.shape[0], int(x.shape[1] / 2), x.shape[2] * 2) 


class PointCloudNet(nn.Module):
    r"""
        PointNet2 as base net with multi-scale grouping
        Point Cloud Upsample Network

        Parameters
        ----------
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        output_channels: int = 6
            Number of output channels.
        num_points: int
            Number of points in point cloud
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, input_channels=3, output_channels=6, use_xyz=True, num_points=1024):
        super(PointCloudNet, self).__init__()
        print(num_points)
        
        
        from ..pointnet import PointNet_features
        
        
         
        self.GLOBAL_module  = PointNet_features(in_dim = 3, dim_k = 256, use_tnet = False, scale = 1)

        num_points = int(num_points)
        self.SA_modules = nn.ModuleList()
        c_in = input_channels
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.2,],
                nsamples=[32,],
                mlps=[[c_in, 32, 32, 64]],
                use_xyz=use_xyz,
            )
        )
        c_out_0 = 64
        num_points = int(num_points / 4)
        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.4,],
                nsamples=[32,],
                mlps=[[c_in, 64, 64, 128]],
                use_xyz=use_xyz,
            )
        )
        c_out_1 = 128
        num_points = int(num_points / 4)
        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.5,],
                nsamples=[32,],
                mlps=[[c_in, 128, 128, 256]],
                use_xyz=use_xyz,
            )
        )
        c_out_2 = 256
        num_points = int(num_points / 4)
        c_in = c_out_2
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.6,],
                nsamples=[32,],
                mlps=[[c_in, 256, 256, 512]],
                use_xyz=use_xyz,
            )
        )
        c_out_3 = 512

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 128, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + c_out_1, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[128 + c_out_0, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels + 3, 128, 256]))
        
        #from ....aggregators.GAP import GAP
        #from ....aggregators.NetVLAD import NetVLADLoupe
        
        self.head_global = GAP(1024, 256)
        self.head_rbf = GAP(512,64)
        self.head_l2 = GAP(64,16)
        
        #self.head2 = NetVLADLoupe(num_clusters = 64, dim = 1024, alpha = 1.0)
    
        
        self.kernel = gaussian
        
        kernels  = 16
        self.RBF_l0_xyz = RBF(3, kernels, self.kernel, init_log_sigmas = 20)
        self.RBF = RBF(1024, kernels, self.kernel, init_log_sigmas = 20)
        self.RBF_l2_xyz = RBF(3, kernels, self.kernel)
        self.RBF_l3_xyz = RBF(256, int(kernels/8), self.kernel)
        self.RBF_l4_xyz = RBF(3, kernels//16, self.kernel)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous()

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        #print(pointcloud.shape)
        num_points = pointcloud.shape[1]
        
        
        #g_features = nn.MaxPool1d(num_points)(self.GLOBAL_module(pointcloud.permute(0, 2, 1)))
        x = self.GLOBAL_module(pointcloud)
        
        #xmean = torch.mean(x, dim=1)
        #x = x - xmean.unsqueeze(1)
        #x = x.unsqueeze(-1)
        #x = x.matmul(x.transpose(2, 1))/(num_points-1)

        features = compute_moments(x.permute(0,2,1),dim=1)
        #dl = x.flatten(start_dim=1)
        # 
        # Averaging over the points
        #x = torch.mean(x, 1) 
        
        #dl = torch.mean(x, dim=-1)
        
        #y = self.RBF(x.permute(0,2,1))
        
        
        #dr = torch.mean(x, dim=1)
        #dl = self.head_global(x)
        
        #return dl
        #xyz, features = self._break_up_pc(pointcloud)
        #l0_xyz, l0_features = xyz, features
        
        #l1_xyz, l1_features = self.SA_modules[0](l0_xyz, l0_features)
        #l2_xyz, l2_features = self.SA_modules[1](l1_xyz, l1_features)
        #dg = self.head_l2(l1_features)
        
        #rfb_features_l2 = self.RBF_l1_xyz(l1_xyz).permute(0,2,1)
        #dr = self.head_rbf(rfb_features_l2)
        
        #l3_xyz, l3_features = self.SA_modules[2](l2_xyz, l2_features)
        #features2 = torch.mean(rfb_features_l2,dim=1)
        
        #l3_xyz, l3_features = self.SA_modules[2](l2_xyz, l2_features)
        #rfb_features_l2 = torch.mean(self.RBF_l3_xyz(l2_features.permute(0,2,1)),dim=1)
        
        #l4_xyz, l4_features = self.SA_modules[3](l3_xyz, l3_features)
        
        #features = torch.mean(l4_features, dim=2)
        #print("Global Features Shape, ", g_features.shape)
        #rfb = torch.cat([dl,dr], dim=1)
        
        return features
        
        #local_features = self.head2(g_features)
        #output = self.fc(torch.cat([g_features, local_features], dim=1))
        

        #l3_features = self.FP_modules[0](l3_xyz, l4_xyz, l3_features, l4_features)
        #l2_features = self.FP_modules[1](l2_xyz, l3_xyz, l2_features, l3_features)
        #l1_features = self.FP_modules[2](l1_xyz, l2_xyz, l1_features, l2_features)
        #l0_features = self.FP_modules[3](l0_xyz, l1_xyz, ip_features, l1_features)
        #print("Local Features Shape, ", l0_features.shape)
        
        #output = {'global': g_features, '1': l0_features, '2': l1_features, '3': l2_features, '4': l3_features}
        #return output
        c_features = torch.cat([g_features, rfb], dim=1)
        #c_features = torch.cat([ip_features, l0_features], dim=1)
        #g_features = g_features.unsqueeze(-1)
        #global_feat = g_features.repeat(1, 1, num_points)
        #c_features = torch.cat([ip_features, l0_features,global_feat], dim=1)
        #print("Concat Features Shape, ", c_features.shape)
        
        return c_features

