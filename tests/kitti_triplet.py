
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.kitti.kitti_triplet import KittiTriplet
from dataloader.laserscan import Scan
from dataloader.sparselaserscan import SparseLaserScan
from torchsparse import SparseTensor
from dataloader.projections import SphericalProjection,BEVProjection

import torch

def test_kitti_pcl_triplet(root):
    modality = Scan()
    dataset = KittiTriplet(root,['00'],modality=modality)
    for i in range(5):
        input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert torch.is_tensor(input['anchor'])
        assert torch.is_tensor(input['positive'][0])
        assert torch.is_tensor(input['negative'][00])
    return True


def test_kitti_sparse_triplet(root):
    sparse = SparseLaserScan(0.05)
    dataset = KittiTriplet(root,['00'],modality=sparse)
    for i in range(5):
        input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert isinstance(input['anchor'],SparseTensor)
        assert isinstance(input['positive'][0],SparseTensor)
        assert isinstance(input['negative'][0],SparseTensor)
    
    return True

def test_kitti_projection_spherical_triplet(root):
    modality = SphericalProjection()
    dataset = KittiTriplet(root,['00'],modality=modality)
    for i in range(5):
        input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert torch.is_tensor(input['anchor'])
        assert torch.is_tensor(input['positive'][0])
        assert torch.is_tensor(input['negative'][00])
    
    return True

def test_kitti_projection_bev_triplet(root):
    modality = BEVProjection(256,256)
    dataset = KittiTriplet(root,['00'],modality=modality)
    for i in range(5):
        input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert torch.is_tensor(input['anchor'])
        assert torch.is_tensor(input['positive'][0])
        assert torch.is_tensor(input['negative'][00])
    
    return True


def run_kitt_triplet_test(root):
    
    print("\n********************************")
    print("KITTI TRIPLET")

    test_kitti_sparse_triplet(root)
    print("[PASSED] Sparse")
    
    test_kitti_pcl_triplet(root)
    print("[PASSED] PCL")

    test_kitti_projection_spherical_triplet(root)
    print("[PASSED] Spherical")

    test_kitti_projection_bev_triplet(root)
    print("[PASSED] BEV")


if __name__=="__main__":

    root = "/home/tiago/Dropbox/research/datasets"
    
    run_kitt_triplet_test(root)