

from dataloader.hortov2.eval import Eval
from torch.utils.data import DataLoader
from dataloader.batch_utils import CollationFunctionFactory
import numpy as np
import torch
import os

# Loader for evaluating new data



class validation_only:
    def __init__(self, **kwargs):
        self.args = kwargs
        

    def get_val_loader(self):
        root  = self.args['root']
        path  = self.args['dataset']['path']
        
        dir = os.path.join(root,path)
        
        sequence  = self.args['dataset']['seq'][0]
        modality = self.args['modality']
        memory = self.args['memory']
        batch_size = self.args['batch_size']
        #print(self.modality)

        if str(modality) in ["bev","spherical","pcl"]:
            self.collation_fn = CollationFunctionFactory("default",voxel_size = 0.05, num_points=10000)
        elif "sparse" in str(modality).lower() :
            self.collation_fn = CollationFunctionFactory("sparse",voxel_size = 0.05, num_points=10000)

        val_loader = Eval( root = dir,
                            sequence = sequence,
                            modality = modality,
                            memory   = memory,
                            )

        valloader  = DataLoader(val_loader,
                                batch_size = batch_size,
                                num_workers= 0,
                                pin_memory=False,
                                collate_fn = self.collation_fn
                                )
        return valloader
    
    def __str__(self):
        return "VALIDATION_ONLY" +'-' + self.args['dataset']['seq'][0]