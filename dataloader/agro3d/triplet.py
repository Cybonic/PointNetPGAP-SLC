

import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import os
import numpy as np
from dataloader.utils import gen_ground_truth
from dataloader.agro3d.agro3d_dataset import agro3d_dataset
from tqdm import tqdm
import pickle

class AGRO3DTriplet():
    def __init__(self,
                 root,
                 sequences,
                 triplet_file,
                 modality = None,
                 memory = 'DISK',
                device = 'cpu'
                    ):
        
        assert modality != None, "Modality does not be None"
        self.modality = modality 

        self.evaluation_mode = False
        self.plc_files  = []
        self.plc_names  = []
        self.poses      = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []
        
        self.row_labels = []
        self.device     = device
        baseline_idx    = 0 
        self.memory = memory 
        
        assert self.memory in ["RAM","DISK"]
        #self.ground_truth_mode = argv['ground_truth']
        assert isinstance(sequences,list)
        for seq in sequences:
            kitti_struct = agro3d_dataset(root, seq)
            
            files,name = kitti_struct._get_point_cloud_file_()
            pose = kitti_struct._get_pose_()
            row_labels = kitti_struct._get_row_labels()

            self.row_labels.extend(row_labels)
            self.plc_files.extend(files)
            self.plc_names.extend(name)
            self.poses.extend(pose)

            triplet_path = os.path.join(root,seq,triplet_file)
            assert os.path.isfile(triplet_path), "Triplet file does not exist " + triplet_path
            
             # load the numpy arrays from the file using pickle
            with open(triplet_path, 'rb') as f:
                data = pickle.load(f)
                seq_anchors   = data['anchors']
                seq_positives = data['positives']
                seq_negatives = data['negatives']
            
            print("Sequence: %s" % seq)
            print("Anchors: %s" % len(seq_anchors))
            for a,p,n in zip(seq_anchors,seq_positives,seq_negatives):
                self.anchors.extend([baseline_idx + a.item()])
                self.positives.extend([baseline_idx + p])
                self.negatives.extend([baseline_idx + n])

            baseline_idx += len(files)

        # Load dataset and laser settings
        self.anchors = np.array(self.anchors)
        self.poses = np.array(self.poses)
        self.row_labels = np.array(self.row_labels)

        self.num_anchors = len(self.anchors)
        self.num_samples = len(self.plc_files)

        n_points = self.poses.shape[0]
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
                
    def load_split(self,split):
        self.anchors = np.array(self.anchors)[split]
        self.num_anchors = len(self.anchors)

    def __len__(self):
        return(self.num_anchors)

    def _get_gt_(self):
        return self.table

    def _get_pose_(self):
        return(self.poses)
    
    def __str__(self):
        return "Triplet_" + str(self.modality)

    def set_evaluation_mode(self):
        self.evaluation_mode = True

    def __getitem__(self,idx):
        # By default return the triplet data
        an_idx,pos_idx,neg_idx  = self.anchors[idx],self.positives[idx], self.negatives[idx]

        if self.memory == "DISK":
            plt_anchor = self.modality(self.plc_files[an_idx])
            plt_pos = [self.modality(self.plc_files[i]) for i in pos_idx]
            plt_neg = [self.modality(self.plc_files[i]) for i in neg_idx]
        else:
            plt_anchor = self.data_on_ram[an_idx]
            plt_pos = [self.data_on_ram[i] for i in pos_idx]
            plt_neg = [self.data_on_ram[i] for i in neg_idx]

        
        pcl =  {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':an_idx,'positive':pos_idx,'negative':neg_idx}

        return(pcl,indx)