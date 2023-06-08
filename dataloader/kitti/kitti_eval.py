
import os
from tqdm import tqdm
import torchvision.transforms as Tr
import numpy as np
from torch.utils.data import DataLoader
from dataloader.kitti.kitti_dataset import kittidataset
from dataloader.utils import gen_ground_truth

PREPROCESSING = Tr.Compose([Tr.ToTensor()])


class KITTIEval:
    def __init__(self,  root, 
                        sequence, 
                        modality = None , 
                        ground_truth = {   
                                        'pos_range':4, # Loop Threshold [m]
                                        'neg_range':10,
                                        'num_neg':20,
                                        'num_pos':1,
                                        'warmupitrs': 600, # Number of frames to ignore at the beguinning
                                        'roi':500}, 
                        device='cpu'):
        
        self.modality = modality
        #self.num_samples = self.num_samples
        self.sequence = sequence
        self.device = device
        kitti_struct = kittidataset(root, 'kitti', sequence)
            
        self.files,name = kitti_struct._get_point_cloud_file_()
        self.poses = kitti_struct._get_pose_()
        self.anchors,self.positives, _ = gen_ground_truth(self.poses,**ground_truth)

        # Load dataset and laser settings
        self.poses = np.array(self.poses)
        self.num_samples = len(self.files)

        n_points = len(self.files)
        self.table = np.zeros((n_points,n_points))
        for a,pos in zip(self.anchors,self.positives):
            for p in pos:
                self.table[a,p]=1

        self.idx_universe = np.arange(self.num_samples)
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)

        
    def __str__(self):
        # seq_name = '_'.join(self.sequence)
        return f'kitti-{self.sequence}'
    
    def get_gt_map(self):
        return(self.table)
    
    def __getitem__(self,index):
        pcl = self.modality(self.files[index])
        return(pcl,index)

    def __len__(self):
        return(len(self.idx_universe))
        
    def get_pose(self):
        pose = self.poses()
        return pose

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)

    def todevice(self,device):
        self.device = device


# ================================================================
#
#
#
#================================================================

class KITTI():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.val_cfg   = kwargs.pop('val_loader')
        self.train_cfg = kwargs['train_loader']
    

    def get_train_loader(self):
        train_loader = KITTITriplet(root = self.root,
                                        **self.train_cfg['data'],
                                        ground_truth = self.train_cfg['ground_truth']
                                                )

        trainloader   = DataLoader(train_loader,
                                batch_size = 1, #train_cfg['batch_size'],
                                shuffle    = self.train_cfg['shuffle'],
                                num_workers= 0,
                                pin_memory=False,
                                drop_last=True,
                                )
        
        return trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    
    def get_val_loader(self):
        val_loader = KITTIEval( root = self.root,
                                **self.val_cfg['data'],
                                ground_truth = self.val_cfg['ground_truth']
                                #**self.kwargs
                                )

        valloader  = DataLoader(val_loader,
                                batch_size = self.val_cfg['batch_size'],
                                num_workers= 0,
                                pin_memory=False,
                                )
        
        return valloader
    
    def get_label_distro(self):
        raise NotImplemented
		#return  1-np.array(self.label_disto)