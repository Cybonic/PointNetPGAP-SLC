import os
import sys
import torch
import torch.nn as nn
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from networks.aggregators.NetVLAD import *
from networks.backbones.pointnet import *
from networks.backbones.spvnas.model_zoo import spvcnn
from networks.pipelines.pipeline_utils import *
from ..backbones import resnet
from networks.utils import *


class PointNetVLAD(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, use_tnet=False, output_dim=1024):
        super(PointNetVLAD, self).__init__()

        self.point_net = PointNet_features(dim_k=feat_dim,use_tnet = use_tnet, scale=1)
        
        self.net_vlad = NetVLADLoupe(feature_size=feat_dim, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        y = self.point_net(x)
        x = self.net_vlad(y)
        return x

    def get_backbone_params(self):
        return self.point_net.parameters()

    def get_classifier_params(self):
        return self.net_vlad.parameters()
  
    def __str__(self):
        return "PointNetVLAD"
    
class ResNet50VLAD(nn.Module):
    def __init__(self,in_dim=3, feat_dim = 1024, num_points=2500, output_dim=1024,**argv):
        super(ResNet50VLAD, self).__init__()

        return_layers = {'layer4': 'out'}
        param = {'pretrained_backbone': False,
                    'out_dim': output_dim,
                    'feat_dim': feat_dim,
                    'in_channels': in_dim,
                    'max_points': num_points,
                    'modality': 'bev'} 

        backbone = resnet.__dict__['resnet50'](param)
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        self.head = NetVLADLoupe(feature_size=feat_dim, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        x = self.backbone(x)
        x = x['out']
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.head(x)
        return x
  
    def __str__(self):
        return "ResNet50VLAD"
    


class SPVVLAD(nn.Module):
    def __init__(self, output_dim=256,feat_dim = 16, num_points=2500, use_tnet=False,**argv):
        super(SPVVLAD, self).__init__()

        self.backbone = spvcnn(output_dim=feat_dim)
        
        self.head = NetVLADLoupe(feature_size=feat_dim, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        
        _, counts = torch.unique(x.C[:, -1], return_counts=True)

        x = self.backbone(x)
        y = torch.split(x, list(counts))
        x = torch.nn.utils.rnn.pad_sequence(list(y)).permute(1, 2,0)
        x = self.head(x)
        return x #, y[:2]

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_classifier_params(self):
        return self.head.parameters()
  
    def __str__(self):
        return "SPVVLAD"
    
    
if __name__ == '__main__':
    _backbone_model_dir = os.path.join(
        os.path.dirname(__file__), '../backbones/spvnas')
    sys.path.append(_backbone_model_dir)
    lidar_pc = np.fromfile(_backbone_model_dir +
                           '/tutorial_data/000000.bin', dtype=np.float32)
    lidar_pc = lidar_pc.reshape(-1, 4)[:,:3]
    input = torch.tensor(lidar_pc).cuda()
    input = input.reshape(1,1,-1, 3) # B,C,N,4
    #input = make_sparse_tensor(lidar_pc, 0.05).cuda()
    batch = torch.cat((input,input))
    model = PointNetVLAD().cuda()
    model.train()
    output = model(batch)
    print('output size: ', output[0].size())
