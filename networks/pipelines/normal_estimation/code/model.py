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
        
        self.GLOBAL_module = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=1),
            #nn.Conv1d(in_channels=input_channels + 3, out_channels=32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
         
        #self.GLOBAL_module  = PointNet_features(in_dim = 3, dim_k = 1024, use_tnet = False, scale = 1)

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
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_points,
                radii=[0.2,],
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
                radii=[0.3,],
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
                radii=[0.4,],
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
        
        from ....aggregators.GAP import GAP
        from ....aggregators.NetVLAD import NetVLADLoupe
        self.head1 = GAP()
        #self.head2 = NetVLADLoupe(num_clusters = 64, dim = 1024, alpha = 1.0)
        
        self.fc = nn.LazyLinear(1024, 256)
        
        self.kernel = gaussian
        
        kernels  = 1024
        self.RBF_l0_xyz = RBF(3, kernels, self.kernel)
        self.RBF_l1_xyz = RBF(3, int(kernels/2), self.kernel)
        self.RBF_l2_xyz = RBF(3, kernels//4, self.kernel)
        self.RBF_l3_xyz = RBF(3, kernels//8, self.kernel)
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
        g_features = self.GLOBAL_module(pointcloud)
        g_features = self.head1(g_features)
        #print("Global Features Shape, ", g_features.shape)
        
        xyz, features = self._break_up_pc(pointcloud)
        l0_xyz, l0_features = xyz, features
        #ip_features = torch.cat((l0_xyz.permute(0, 2, 1), l0_features), dim=1)
        
        rfb_features_l0 = torch.mean(self.RBF_l0_xyz(l0_xyz), dim=1)
        
        l1_xyz, l1_features = self.SA_modules[0](l0_xyz, l0_features)
        rfb_features_l1 = torch.mean(self.RBF_l1_xyz(l1_xyz),dim=1)
        
        l2_xyz, l2_features = self.SA_modules[1](l1_xyz, l1_features)
        rfb_features_l2 = torch.mean(self.RBF_l2_xyz(l2_xyz),dim=1)
        
        l3_xyz, l3_features = self.SA_modules[2](l2_xyz, l2_features)
        rfb_features_l3 = torch.mean(self.RBF_l3_xyz(l3_xyz),dim=1)
        
        l4_xyz, l4_features = self.SA_modules[3](l3_xyz, l3_features)
        rfb_features_l4 = torch.mean(self.RBF_l4_xyz(l4_xyz),dim=1)
        
        features = torch.mean(l4_features, dim=2)
        rfb = torch.cat([rfb_features_l2,rfb_features_l3,rfb_features_l4], dim=1)
        
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

