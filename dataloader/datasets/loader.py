

from dataloader.datasets.eval import Eval
from dataloader.datasets.triplet import Triplet
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
        dataset = self.train_cfg['dataset']
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
                                    dataset     = dataset,
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
        dataset  = self.val_cfg['dataset']
        sequence  = self.val_cfg['sequence']
        ground_truth_files = self.val_cfg['ground_truth_file']
        augmentation = self.val_cfg['augmentation']
        
        #print(self.modality)

        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
        elif "sparse" in str(self.modality).lower() :
            self.collation_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)

        val_loader = Eval( root = self.root,
                                dataset = dataset,
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

