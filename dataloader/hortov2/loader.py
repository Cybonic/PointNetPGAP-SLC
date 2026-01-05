

from dataloader.new_datasets.eval import Eval
from torch.utils.data import DataLoader
from dataloader.batch_utils import CollationFunctionFactory
import numpy as np
import torch

# Loader for evaluating new data

class evaluation_on_new_data():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.val_cfg   = kwargs.pop('val_loader')
        self.modality  = kwargs['modality']
        self.max_points = kwargs['max_points']
        self.memory = kwargs['memory']

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
                                sequence = sequence,
                                modality = self.modality,
                                memory= self.memory,
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
        return "EVALUATION_ON_NEW_DATA"
