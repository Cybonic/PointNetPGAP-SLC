

import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import os
import numpy as np
from dataloader.utils import gen_ground_truth
from dataloader.kitti.kitti_dataset import kittidataset



class KittiTriplet():
    def __init__(self,
                 root,
                 sequences,
                 modality=None,
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
        #self.ground_truth_mode = argv['ground_truth']
        assert isinstance(sequences,list)

        for seq in sequences:
            kitti_struct = kittidataset(root, 'kitti', seq)
            
            files,name = kitti_struct._get_point_cloud_file_()
            
            self.plc_files.extend(files)
            self.plc_names.extend(name)
            
            pose = kitti_struct._get_pose_()
            self.poses.extend(pose)
            
            anchors,positives,negatives = gen_ground_truth(pose,**ground_truth)
            
            self.anchors.extend(baseline_idx + np.array(anchors))
            self.positives.extend(baseline_idx + np.array(positives))
            self.negatives.extend(baseline_idx + np.array(negatives))

            baseline_idx += len(files)

        # Load dataset and laser settings
        self.anchors = np.array(self.anchors)
        self.poses = np.array(self.poses)

        self.num_samples = len(self.anchors)

        n_points = baseline_idx
        self.table = np.zeros((n_points,n_points))
        for a,pos in zip(self.anchors,self.positives):
            for p in pos:
                self.table[a,p]=1
    

    def __len__(self):
        return(self.num_samples)

    def _get_gt_(self):
        return self.table

    def _get_pose_(self):
        return(self.poses)
    
    def __str__(self):
        return "Kitti"
    
    def __getitem__(self,idx):
        an_idx,pos_idx,neg_idx  = self.anchors[idx],self.positives[idx], self.negatives[idx]

        plt_anchor = self.modality(self.plc_files[an_idx])
        plt_pos = [self.modality(self.plc_files[i]) for i in pos_idx]
        plt_neg = [self.modality(self.plc_files[i]) for i in neg_idx]

        pcl = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':an_idx,'positive':len(pos_idx),'negative':len(neg_idx)}

        return(pcl,indx)
