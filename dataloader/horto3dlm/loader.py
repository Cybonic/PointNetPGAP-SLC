

from dataloader.horto3dlm.eval import Eval
from dataloader.horto3dlm.triplet import Triplet
from torch.utils.data import DataLoader,SubsetRandomSampler
from dataloader.batch_utils import CollationFunctionFactory
import numpy as np
import torch

class cross_validation():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.dataset = kwargs.pop('dataset')
        self.val_cfg   = kwargs.pop('val_loader')
        self.train_cfg = kwargs['train_loader']
        self.modality  = kwargs['modality']
        self.max_points = kwargs['max_points']
        self.memory = kwargs['memory']

        

    def get_train_loader(self,debug=False):
        sequence  = self.train_cfg['sequence']
        triplet_files = self.train_cfg['triplet_file']
        augmentation = self.train_cfg['augmentation'] if 'augmentation' in self.train_cfg else 0
        shuffle_points = self.train_cfg['shuffle_points'] if 'shuffle_points' in self.train_cfg else 0
   
        
        #max_points = self.max_points
        print(self.modality)
        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("torch_tuple",voxel_size = 0.05, num_points=10000)
        elif "sparse"in str(self.modality).lower():
            self.collation_fn = CollationFunctionFactory("sparse_tuple",voxel_size = 0.05, num_points=10000)

        train_loader = Triplet(root       = self.root,
                                    dataset     = self.dataset,
                                    sequences   = sequence,
                                    triplet_file = triplet_files,
                                    modality = self.modality,
                                    #ground_truth = self.train_cfg['ground_truth'],
                                    memory= self.memory,
                                    augmentation = augmentation,
                                    shuffle_points = shuffle_points
                                    
                                                )
        
        if debug == False:
            trainloader   = DataLoader(train_loader,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = self.train_cfg['shuffle'],
                                    num_workers= 0,
                                    pin_memory =False,
                                    drop_last  =True,
                                    collate_fn = self.collation_fn
                                    )
        else:
            indices = np.random.randint(0,len(train_loader),20)
            np.random.shuffle(indices)
            sampler = SubsetRandomSampler(indices)

            trainloader   = DataLoader(train_loader,
                                    batch_size  = 1, #train_cfg['batch_size'],
                                    shuffle     = False,
                                    num_workers = 0,
                                    pin_memory  =False,
                                    drop_last   =True,
                                    sampler=sampler,
                                    collate_fn = self.collation_fn
                                    )
        
        return trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    
    def get_val_loader(self):
        sequence  = self.val_cfg['sequence']
        ground_truth_files = self.val_cfg['ground_truth_file']
        augmentation = self.val_cfg['augmentation']
        
        #print(self.modality)

        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
        elif "sparse" in str(self.modality).lower() :
            self.collation_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)

        val_loader = Eval( root = self.root,
                                dataset = self.dataset,
                                sequence = sequence[0],
                                modality = self.modality,
                                memory= self.memory,
                                ground_truth_file = ground_truth_files,
                                augmentation = augmentation
                                )

        valloader  = DataLoader(val_loader,
                                batch_size = self.val_cfg['batch_size'],
                                num_workers= 0,
                                pin_memory=False,
                                collate_fn = self.collation_fn
                                )
        return valloader
    
    def __str__(self):
        return "CROSS_VALIDATION"
		#return  1-np.array(self.label_disto)



"""
================================================================================================
================================================================================================
"""



class split():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.dataset = kwargs.pop('dataset')
        self.val_cfg   = kwargs.pop('val_loader')
        self.train_cfg = kwargs['train_loader']
        self.modality  = kwargs['modality']
        self.max_points = kwargs['max_points']
        self.memory = kwargs['memory']

        self.split_ratio = 0.8
        if "split_ratio" in kwargs.keys():
            self.split_ratio = kwargs['split_ratio']

        sequence  = self.train_cfg['sequence']
        triplet_files = self.train_cfg['triplet_file']

        #max_points = self.max_points
        print(self.modality)
        
        self.train_set = KittiTriplet( root = self.root,
                                    dataset     = self.dataset,
                                    sequences   = sequence,
                                    triplet_file = triplet_files,
                                    modality = self.modality,
                                    #ground_truth = self.train_cfg['ground_truth'],
                                    memory= self.memory
                                    )
        
        num_anchors = len(self.train_set)

        sequence  = self.val_cfg['sequence']
        ground_truth_files = self.val_cfg['ground_truth_file']

        self.val_set = KITTIEval(   root     = self.root,
                                    dataset  = self.dataset,
                                    sequence = sequence[0],
                                    modality = self.modality,
                                    memory   = self.memory,
                                    ground_truth_file = ground_truth_files
                                )
        
        # Split the dataset into train and test
        indices = list(range(num_anchors))
        np.random.shuffle(indices)
        self.split_val_index = int(np.floor(self.split_ratio * num_anchors))
        self.train_indices, self.val_indices = indices[:self.split_val_index], indices[self.split_val_index:]
        
        self.val_set.load_split(self.val_indices)   
        self.train_set.load_split(self.train_indices)

        
    def get_train_loader(self):
        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("torch_tuple",voxel_size = 0.05, num_points=10000)
        elif "sparse"in str(self.modality).lower():
            self.collation_fn = CollationFunctionFactory("sparse_tuple",voxel_size = 0.05, num_points=10000)

        trainloader   = DataLoader(self.train_set,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = self.train_cfg['shuffle'],
                                    num_workers= 0,
                                    pin_memory = False,
                                    drop_last  = True,
                                    collate_fn = self.collation_fn
                                    )
        return trainloader
    
    def get_val_loader(self):
         
        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
        elif "sparse" in str(self.modality).lower() :
            self.collation_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)

        valloader  = DataLoader(self.val_set,
                                batch_size = self.val_cfg['batch_size'],
                                num_workers= 0,
                                pin_memory=False,
                                collate_fn = self.collation_fn,
                                #sampler= SubsetRandomSampler(self.val_indices)
                                )
        return valloader
    
    def __str__(self):
        return "SPLIT"