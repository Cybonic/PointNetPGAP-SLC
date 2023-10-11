
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.kitti.kitti_triplet import KittiTriplet
from dataloader.laserscan import Scan
from dataloader.sparselaserscan import SparseLaserScan
from torchsparse import SparseTensor
from dataloader.projections import SphericalProjection,BEVProjection
import numpy as np
import torch
import tqdm
def run_test_on_dataloader(dataloader,ground_truth,poses):
    
    for input,idx in tqdm.tqdm(dataloader,"TEST | TENSOR"):
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
    print("PASSED| Tensor test ")

    for input,idx in tqdm.tqdm(dataloader,"TEST | DIMENSIONALITY"):
        # Check dimensionality
        assert len(positive) == ground_truth['num_pos']
        assert len(negative) == ground_truth['num_neg']
    print("PASSED| Dimensionality test ")

    for input,idx in tqdm.tqdm(dataloader,"TEST | RANGE"):
        # Check range
        an_pose  = poses[idx['anchor']]
        pos_pose = poses[idx['positive']]
        neg_pose = poses[idx['negative']]
        # Compute ranges
        pos_range = np.linalg.norm(an_pose-pos_pose, axis=-1)
        neg_range = np.linalg.norm(an_pose-neg_pose, axis=-1)

        assert all(pos_range <= float(ground_truth['pos_range']))
        assert all(neg_range >= float(ground_truth['neg_range'])),f"{ float(ground_truth['neg_range'])} < " + ' '.join([str(round(value,1)) for value in neg_range])
    print("PASSED| Range test ")

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



def test_kitti_pcl_triplet_batching(root,memory="DISK"):
    
    ground_truth = { 'pos_range':1, # Loop Threshold [m]
                    'neg_range':50,
                    'num_neg':20,
                    'num_pos':5,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500
                    }
    
    modality = Scan(max_points=10000)
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
            
    collat_fn = CollationFunctionFactory("torch_tuple",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    poses = dataset.poses
    run_test_on_dataloader(dataloader,ground_truth,poses)
    return True



def test_kitti_spherical_projection_triplet_batching(root,memory="DISK"):
    
    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':5,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500
                    }
    
    modality = SphericalProjection()
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
            
    collat_fn = CollationFunctionFactory("torch_tuple",voxel_size = 0.05, num_points=10000)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn= collat_fn,
                            sampler=sampler)
    
    poses = dataset.poses
    run_test_on_dataloader(dataloader,ground_truth,poses)

    return True




def test_kitti_pcl_triplet_groundtruth(root,memory="DISK"):
    
    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':5,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500
                    }
    
    modality = Scan(max_points=10000)
    dataset = KittiTriplet( root,['00'],
                            modality=modality,
                            ground_truth = ground_truth,
                            debug = True,
                            memory=memory,
                            )
        
    poses = dataset.poses
    run_test_on_dataloader(dataset,ground_truth,poses)

    return True


def test_kitti_pcl_triplet(root,memory="DISK"):
    modality = Scan()
    dataset = KittiTriplet(root,['00'],
                           modality=modality,
                           memory=memory)
    
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


def test_kitti_sparse_triplet(root,memory="DISK"):
    modality = SparseLaserScan(0.05)
    dataset = KittiTriplet(root,['00'],
                           modality=modality,
                           debug=True,
                           memory=memory)
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

def test_kitti_projection_spherical_triplet(root,memory="DISK"):
    modality = SphericalProjection()
    dataset = KittiTriplet(root,['00'],
                           modality=modality,
                           debug=True,
                           memory=memory)
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

def test_kitti_projection_bev_triplet(root,memory="DISK"):
    modality = BEVProjection(256,256)
    dataset = KittiTriplet(root,['00'],
                            modality=modality,
                            debug=True,
                            memory=memory)
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



def test_kitti_pcl_triplet_from_file(root,dataset,sequence,triplet_file,memory="DISK"):


    files = os.path.join(root,dataset,sequence[0],triplet_file)

    modality = Scan(max_points=10000)
    
    dataset = KittiTriplet( root,
                            dataset,
                            sequence,
                            modality = modality,
                            triplet_file = files,
                            memory = memory,
                            )
        
    poses = dataset.poses

    ground_truth = { 'pos_range':2, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':1
                    }


    run_test_on_dataloader(dataset,ground_truth,poses)

    return True



if __name__=="__main__":

    #root = "/home/tiago/Dropbox/SHARE/DATASET"
    root = "/home/deep/Dropbox/SHARE/DATASET"
    dataset = "uk"
    sequence = ["orchards/june23/extracted"]
    triplet_file = "ground_truth_ar1m_nr10m_pr2m.pkl"

    test_kitti_pcl_triplet_from_file(root,dataset,sequence,triplet_file, memory="DISK")

    #run_kitt_triplet_test(root)
    
    #test_kitti_spherical_projection_triplet_batching(root)
    #test_kitti_pcl_triplet_batching(root)
    #test_kitti_sparse_triplet_batching(root)
    #run_kitt_triplet_test(root)