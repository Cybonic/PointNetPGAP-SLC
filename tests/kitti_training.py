
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.kitti.kitti_triplet import KittiTriplet
from dataloader.laserscan import Scan
from dataloader.sparselaserscan import SparseLaserScan
from torchsparse import SparseTensor
from dataloader.projections import SphericalProjection,BEVProjection
import numpy as np
import torch

def run_test_on_tensor_dataloader(dataloader,ground_truth,poses):
    for input,idx in dataloader:
        #input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys

        positive = input['positive']
        negative = input['negative']

        assert torch.is_tensor(input['anchor'])
        assert torch.is_tensor(positive[0])
        assert torch.is_tensor(negative[00])
        # Check dimensionality
        assert len(positive) == ground_truth['num_pos']
        assert len(negative) == ground_truth['num_neg']

        # Check range
        an_pose  = poses[idx['anchor']]
        pos_pose = poses[idx['positive']]
        neg_pose = poses[idx['negative']]
        # Compute ranges
        pos_range = np.linalg.norm(an_pose-pos_pose, axis=-1)
        neg_range = np.linalg.norm(an_pose-neg_pose, axis=-1)

        assert all(pos_range < ground_truth['pos_range'])
        assert all(neg_range > ground_truth['neg_range'])

    return True



def run_test_on_sparse_dataloader(dataloader,ground_truth,poses):
    for input,idx in dataloader:
        #input,idx = dataset[i]
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys

        positive = input['positive']
        negative = input['negative']

        assert isinstance(input['anchor'],SparseTensor)
        assert isinstance(positive[0],SparseTensor)
        assert isinstance(negative[00],SparseTensor)

        # Check dimensionality
        assert len(positive) == ground_truth['num_pos']
        assert len(negative) == ground_truth['num_neg']

        # Check range
        an_pose  = poses[idx['anchor']]
        pos_pose = poses[idx['positive']]
        neg_pose = poses[idx['negative']]
        # Compute ranges
        pos_range = np.linalg.norm(an_pose-pos_pose, axis=-1)
        neg_range = np.linalg.norm(an_pose-neg_pose, axis=-1)

        assert all(pos_range < ground_truth['pos_range'])
        assert all(neg_range > ground_truth['neg_range'])

    return True



def test_kitti_sparse_triplet_batching(root,memory="DISK"):
    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':5,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500
                    }
    modality = SparseLaserScan(0.05)
    dataset = KittiTriplet( root,['00'],
                            modality=modality,
                            ground_truth = ground_truth,
                            debug = True,
                            memory=memory,
                            )
    
    from torch.utils.data import DataLoader,SubsetRandomSampler 
    from dataloader.utils import CollationFunctionFactory
    import numpy as np
    
    indices = np.random.randint(0,len(dataset),5)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
            
    collat_fn = CollationFunctionFactory("sparse_tuple",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    for input,idx in dataloader:
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert isinstance(input['anchor'],SparseTensor)
        assert isinstance(input['positive'],SparseTensor)
        assert isinstance(input['negative'],SparseTensor)

        #assert torch.is_tensor(input)
    return True


def test_kitti_sparse_training(root,memory="DISK"):
    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':5,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500
                    }
    
    

    modality = SparseLaserScan(0.05)
    dataset = KittiTriplet( root,['00'],
                            modality=modality,
                            ground_truth = ground_truth,
                            debug = True,
                            memory=memory,
                            )
    
    from torch.utils.data import DataLoader,SubsetRandomSampler 
    from dataloader.utils import CollationFunctionFactory
    import numpy as np
    
    indices = np.random.randint(0,len(dataset),5)
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
            
    collat_fn = CollationFunctionFactory("sparse_tuple",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    from networks.network_pipeline import get_pipeline
    model = get_pipeline("LOGG3D", num_points=4096,output_dim=256)
    
    for input,idx in dataloader:
        keys = list(input.keys())
        assert 'anchor'   in keys
        assert 'positive' in keys
        assert 'negative' in keys
        assert isinstance(input['anchor'],SparseTensor)
        assert isinstance(input['positive'],SparseTensor)
        assert isinstance(input['negative'],SparseTensor)

        


        model()

        #assert torch.is_tensor(input)
    return True










def run_kitt_triplet_test(root):
    
    print("\n********************************")
    print("KITTI TRIPLET")

    test_kitti_pcl_triplet_groundtruth(root)
    print("[PASSED] PCL groundtruth")
    #test_kitti_pcl_triplet_groundtruth(root,memory="RAM")
    #print("[PASSED] PCL groundtruth RAM")

    test_kitti_sparse_triplet(root)
    print("[PASSED] Sparse")
    test_kitti_sparse_triplet(root,memory="RAM")
    print("[PASSED] Sparse RAM")
    
    test_kitti_pcl_triplet(root)
    print("[PASSED] PCL")
    test_kitti_pcl_triplet(root,memory="RAM")
    print("[PASSED] PCL RAM")

    test_kitti_projection_spherical_triplet(root)
    print("[PASSED] Spherical")
    test_kitti_projection_spherical_triplet(root,memory="RAM")
    print("[PASSED] Spherical RAM")

    test_kitti_projection_bev_triplet(root)
    print("[PASSED] BEV")

    test_kitti_projection_bev_triplet(root,memory="RAM")
    print("[PASSED] BEV RAM")

    test_kitti_sparse_triplet_batching(root)
    print("[PASSED] sparse batching RAM")

    test_kitti_pcl_triplet_batching(root)
    print("[PASSED] pcl batching RAM")

    test_kitti_spherical_projection_triplet_batching(root)
    print("[PASSED] Spherical batching RAM")
if __name__=="__main__":

    root = "/home/tiago/Dropbox/research/datasets"
    root = "/media/deep/datasets/datasets"
    
    test_kitti_sparse_training(root,memory="DISK")

    #run_kitt_triplet_test(root)
    
    #test_kitti_spherical_projection_triplet_batching(root)
    #test_kitti_pcl_triplet_batching(root)
    #test_kitti_sparse_triplet_batching(root)
    #run_kitt_triplet_test(root)