
import os,sys
from tqdm import tqdm
import torchvision.transforms as Tr
from dataloader.utils import extract_points_in_rectangle_roi
from dataloader.utils import rotate_poses
from dataloader import row_segments

import numpy as np

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from dataloader.kitti.kitti_dataset import kittidataset

import pickle

PREPROCESSING = Tr.Compose([Tr.ToTensor()])


class KITTIEval:
    def __init__(self,  root, 
                        dataset,
                        sequence, 
                        ground_truth_file,
                        modality = None ,
                        memory= "DISK", 
                        debug = False,
                        
                        device='cpu'):
        
        assert memory in ["RAM", "DISK"]
        self.memory   = memory 
        self.modality = modality
        #self.num_samples = self.num_samples
        self.sequence = sequence
        self.device   = device
        kitti_struct = kittidataset(root,dataset, sequence)
            
        self.files,name = kitti_struct._get_point_cloud_file_()
        self.poses = kitti_struct._get_pose_()
        
        dirs = sequence.split('/')
        seq  = '_'.join(dirs[:2]) #sequence.replace('/','_')
        assert seq in row_segments.__dict__.keys(), "Sequence not found in row_segments.py"
        dataset_raws = row_segments.__dict__[seq]
        # Load aline rotation
        # Rotate poses to match the image frame
        xy = rotate_poses(self.poses.copy(),dataset_raws['angle'])
        rectangle_rois = np.array(dataset_raws['rows'])

        self.row_labels  = extract_points_in_rectangle_roi(xy,rectangle_rois)

        ground_truth_path = os.path.join(root,dataset,sequence,ground_truth_file)
        assert os.path.isfile(ground_truth_path), "Ground truth file does not exist " + ground_truth_path
        # Load dataset and laser settings
        self.poses = np.array(self.poses)
        self.num_samples = len(self.files)
        
        if debug == True:
            self.set_debug()
        
        with open(ground_truth_path, 'rb') as f:
            data = pickle.load(f)
            self.anchors   = data['anchors']
            self.positives = data['positives']

        # self.anchors,self.positives, _ = gen_ground_truth(self.poses,**ground_truth)

        n_points = len(self.files)
        self.table = np.zeros((n_points,n_points))
        for a,pos in zip(self.anchors,self.positives):
            for p in pos:
                self.table[a,p]=1

        if self.memory == "RAM":
            self.load_to_RAM()

        self.idx_universe = np.arange(self.num_samples)
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)

   
    def load_to_RAM(self):
        self.memory=="RAM"
        indices = list(range(self.num_samples))
        self.data_on_ram = []
        for idx in tqdm(indices,"Load to RAM"):
            plt = self.modality(self.files[idx])
            self.data_on_ram.append(plt)

    def set_debug(self):
        indices = np.random.randint(0,self.num_samples,20)
        self.files = self.files[indices]
        self.poses = self.poses[indices]
        self.num_samples = len(indices) # Update number of files

    def __str__(self):
        return f'kitti-{self.sequence}'
    
    def get_gt_map(self):
        return(self.table)
    
    def __getitem__(self,index):
        
        if self.memory=="RAM":
            pcl = self.data_on_ram[index]
        else:
            pcl = self.modality(self.files[index]).long()

        return(pcl,index)

    def __len__(self):
        return(len(self.idx_universe))
        
    def get_pose(self):
        return self.poses

    def get_row_labels(self):
        return self.row_labels
    
    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)

    def todevice(self,device):
        self.device = device

