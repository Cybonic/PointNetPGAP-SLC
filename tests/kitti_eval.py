
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.kitti.kitti_eval import KITTIEval
from dataloader.laserscan import Scan
from dataloader.sparselaserscan import SparseLaserScan
from torchsparse import SparseTensor
from dataloader.projections import SphericalProjection,BEVProjection

import torch

def test_kitti_pcl_eval(root,max_points=10000,memory="DISK"):
    modality = Scan(max_points=max_points)
    dataset = KITTIEval(root,'00',
                        modality=modality,
                        memory=memory,
                        debug=True)
        
    for i in range(5):
        input,idx = dataset[i]
        n,f = input.shape
        assert max_points == n
        assert torch.is_tensor(input)
    return True


def test_kitti_sparse_eval(root,memory="DISK"):
    modality = SparseLaserScan(0.05)
    dataset = KITTIEval(root,'00',
                        modality=modality,
                        memory=memory,
                        debug=True)
    
    for i in range(5):
        input,idx = dataset[i]
        assert isinstance(input,SparseTensor)
    
    return True

def test_kitti_projection_spherical_eval(root,memory="DISK"):
    modality = SphericalProjection()
    dataset = KITTIEval(root,'00',
                        modality=modality,
                        memory=memory,
                        debug=True)
        
    for i in range(5):
        input,idx = dataset[i]
        assert torch.is_tensor(input)
    
    return True

def test_kitti_projection_bev_eval(root,memory="DISK"):
    modality = BEVProjection(256,256)
    dataset = KITTIEval(root,'00',
                        modality=modality,
                        memory=memory,
                        debug=True)
        
    for i in range(5):
        input,idx = dataset[i]
        assert torch.is_tensor(input)
    return True

def test_kitti_batching_sparse(root,memory="DISK"):
    modality = SparseLaserScan(0.05)
    dataset = KITTIEval(root,'00', 
                        modality=modality, 
                        memory = memory, # RAM or Disk
                        debug = True)
    
    from torch.utils.data import DataLoader,SubsetRandomSampler 
    from dataloader.utils import CollationFunctionFactory
    import numpy as np
    
    indices = np.random.randint(0,len(dataset),5)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
            
    collat_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    for input,idx in dataloader:
        assert isinstance(input,SparseTensor)
        assert isinstance(idx,list)

        #assert torch.is_tensor(input)
    return True

def test_kitti_batching_pts_torch(root,memory="DISK"):
    modality = Scan(max_points=1000)
    dataset = KITTIEval(root,'00', 
                        modality=modality, 
                        memory = memory, # RAM or Disk
                        debug = True)
    
    from torch.utils.data import DataLoader,SubsetRandomSampler 
    from dataloader.utils import CollationFunctionFactory
    import numpy as np
    
    indices = np.random.randint(0,len(dataset),5)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
            
    collat_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    for input,idx in dataloader:
        assert isinstance(input,torch.Tensor)
        assert isinstance(idx,list)
        
        #assert torch.is_tensor(input)
    return True


def test_kitti_batching_prj_torch(root,memory="DISK"):
    modality = BEVProjection(256,256)
    dataset = KITTIEval(root,'00', 
                        modality=modality, 
                        memory = memory, # RAM or Disk
                        debug = True)
    
    from torch.utils.data import DataLoader,SubsetRandomSampler 
    from dataloader.utils import CollationFunctionFactory
    import numpy as np
    
    indices = np.random.randint(0,len(dataset),5)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
            
    collat_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    for input,idx in dataloader:
        assert isinstance(input,torch.Tensor)
        assert isinstance(idx,list)
        
        #assert torch.is_tensor(input)
    return True

def run_kitt_eval_test(root):
    print("\n********************************")
    print("KITTI EVAL")

    #test_kitti_pcl_eval(root,memory="DISK")
    #print("[PASSED] PCL DISK")
    test_kitti_pcl_eval(root,memory="RAM")
    print("[PASSED] PCL RAM")
    
    test_kitti_sparse_eval(root,memory="DISK")
    print("[PASSED] SPARSE DISK")
    test_kitti_sparse_eval(root,memory="RAM")
    print("[PASSED] SPARSE RAM")

    test_kitti_projection_spherical_eval(root,memory="DISK")
    print("[PASSED] Spherical DISK")
    test_kitti_projection_spherical_eval(root,memory="RAM")
    print("[PASSED] Spherical RAM")

    test_kitti_projection_bev_eval(root,memory="DISK")
    print("[PASSED] BEV DISK")
    test_kitti_projection_bev_eval(root,memory="RAM")
    print("[PASSED] BEV RAM")
    
    test_kitti_batching_sparse(root,memory="DISK")
    print("[PASSED] SPARSE BATCHING DISK")
    
    test_kitti_batching_pts_torch(root,memory="DISK")
    print("[PASSED] PCLs BATCHING DISK")
    
    test_kitti_batching_prj_torch(root)
    print("[PASSED] PROJ BATCHING DISK")
    


    

if __name__=="__main__":

    root = "/home/tiago/Dropbox/research/datasets"
    print("\n************************************************")
    print("Kitti Eval Test\n")
    #test_kitti_batching_sparse(root,memory="DISK")
    
    #test_kitti_batching_pts_torch(root,memory="DISK")
    test_kitti_batching_prj_torch(root)
    #test_kitti_pcl_eval(root)
    #print("Sparse passed!")
    
    #test_kitti_sparse_eval(root)
    #print("PCL    passed!")

    #test_kitti_projection_spherical_eval(root)
    #print("Spherical passed!")

    #test_kitti_projection_bev_eval(root)
    #print("BEV passed!")
    #print("************************************************\n")