

import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import os
import numpy as np
from dataloader.utils import gen_ground_truth
from dataloader.kitti.kitti_dataset import kittidataset
import torch
from tqdm import tqdm

class KittiTriplet():
    def __init__(self,
                 root,
                 sequences,
                 modality = None,
                 memory = 'DISK',
                 debug = False,
                 ground_truth = {   'pos_range':4, # Loop Threshold [m]
                                    'neg_range':10,
                                    'num_neg':20,
                                    'num_pos':1,
                                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                                    'roi':500},
                device = 'cpu'
                    ):
        
        assert modality != None, "Modality does not be None"
        self.modality = modality 

        self.plc_files  = []
        self.plc_names  = []
        self.poses      = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []
        self.device     = device
        baseline_idx  = 0 
        self.memory = memory 
        assert self.memory in ["RAM","DISK"]
        #self.ground_truth_mode = argv['ground_truth']
        assert isinstance(sequences,list)

        for seq in sequences:
            kitti_struct = kittidataset(root, 'kitti', seq)
            
            files,name = kitti_struct._get_point_cloud_file_()
            pose = kitti_struct._get_pose_()
                
        
            self.plc_files.extend(files)
            self.plc_names.extend(name)
            self.poses.extend(pose)
            
            anchors,positives,negatives = gen_ground_truth(pose,**ground_truth)
            

            self.anchors.extend(baseline_idx + np.array(anchors))
            self.positives.extend(baseline_idx + np.array(positives))
            self.negatives.extend(baseline_idx + np.array(negatives))

            baseline_idx += len(files)

        # Load dataset and laser settings
        self.anchors = np.array(self.anchors)
        self.poses = np.array(self.poses)

        self.num_anchors = len(self.anchors)
        self.num_samples = len(self.plc_files)

        n_points = baseline_idx
        self.table = np.zeros((n_points,n_points))
        for a,pos in zip(self.anchors,self.positives):
            for p in pos:
                self.table[a,p]=1
    
        # Load Data to RAM
        if self.memory == 'RAM':
            self.load_to_RAM()

    def load_to_RAM(self):
        self.memory=="RAM"
        indices = list(range(self.num_samples))
        self.data_on_ram = []
        for idx in tqdm(indices,"Load to RAM"):
            plt = self.modality(self.plc_files[idx])
            self.data_on_ram.append(plt)
                

    def __len__(self):
        return(self.num_anchors)

    def _get_gt_(self):
        return self.table

    def _get_pose_(self):
        return(self.poses)
    
    def __str__(self):
        return "Kitti"
    
    def __getitem__(self,idx):
        an_idx,pos_idx,neg_idx  = self.anchors[idx],self.positives[idx], self.negatives[idx]

        if self.memory == "DISK":
            plt_anchor = self.modality(self.plc_files[an_idx])
            plt_pos = [self.modality(self.plc_files[i]) for i in pos_idx]
            plt_neg = [self.modality(self.plc_files[i]) for i in neg_idx]
        else:
            plt_anchor = self.data_on_ram[an_idx]
            plt_pos = [self.data_on_ram[i] for i in pos_idx]
            plt_neg = [self.data_on_ram[i] for i in neg_idx]

            
        pcl = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':an_idx,'positive':pos_idx,'negative':neg_idx}

        return(pcl,indx)