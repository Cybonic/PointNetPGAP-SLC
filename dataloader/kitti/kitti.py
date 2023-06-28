

from dataloader.kitti.kitti_eval import KITTIEval
from dataloader.kitti.kitti_triplet import KittiTriplet
from torch.utils.data import DataLoader,SubsetRandomSampler
from dataloader.utils import CollationFunctionFactory
import numpy as np

class KITTI():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.val_cfg   = kwargs.pop('val_loader')
        self.train_cfg = kwargs['train_loader']
        self.modality  = kwargs['modality']
        self.max_points = kwargs['max_points']
        self.memory = kwargs['memory']

        

    def get_train_loader(self,debug=False):
        sequence  = self.train_cfg['sequence']
        #max_points = self.max_points
        print(self.modality)
        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("torch_tuple",voxel_size = 0.05, num_points=10000)
        elif str(self.modality) in  "sparse":
            self.collation_fn = CollationFunctionFactory("sparse_tuple",voxel_size = 0.05, num_points=10000)

        train_loader = KittiTriplet( root = self.root,
                                    sequences = sequence,
                                    modality = self.modality,
                                    ground_truth = self.train_cfg['ground_truth'],
                                    memory= self.memory
                                                )
        
        if debug == False:
            trainloader   = DataLoader(train_loader,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = self.train_cfg['shuffle'],
                                    num_workers= 0,
                                    pin_memory=False,
                                    drop_last=True,
                                    collate_fn = self.collation_fn
                                    )
        else:
            indices = np.random.randint(0,len(train_loader),20)
            np.random.shuffle(indices)
            sampler = SubsetRandomSampler(indices)

            trainloader   = DataLoader(train_loader,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = False,
                                    num_workers= 0,
                                    pin_memory=False,
                                    drop_last=True,
                                    sampler=sampler,
                                    collate_fn = self.collation_fn
                                    )
        
        return trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    
    def get_val_loader(self):
        sequence  = self.val_cfg['sequence']
        print(self.modality)
        if str(self.modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
        elif str(self.modality) in  "sparse":
            self.collation_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)

        val_loader = KITTIEval( root = self.root,
                               sequence = sequence[0],
                               modality = self.modality,
                                memory= self.memory,
                                ground_truth = self.val_cfg['ground_truth']
                                )

        valloader  = DataLoader(val_loader,
                                batch_size = self.val_cfg['batch_size'],
                                num_workers= 0,
                                pin_memory=False,
                                collate_fn = self.collation_fn
                                )
        
        return valloader
    
    def __str__(self):
        return 'kitti'
		#return  1-np.array(self.label_disto)