
import os,sys
from tqdm import tqdm
import torchvision.transforms as Tr
from dataloader.utils import extract_points_in_rectangle_roi
from dataloader.utils import rotate_poses

import numpy as np

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from dataloader.new_datasets.dataset import file_structure

import pickle

PREPROCESSING = Tr.Compose([Tr.ToTensor()])


class Eval:
    def __init__(self,  root, 
                        sequence,
                        modality = None ,
                        memory= "DISK", 
                        debug = False,
                        device='cpu',
                        augmentation = False
                        ):
        
        assert memory in ["RAM", "DISK"]
        self.memory   = memory 
        self.modality = modality
        self.augmentation = bool(augmentation)
        self.sequence = sequence
        
        #self.num_samples = self.num_samples
        self.device   = device
        kitti_struct = file_structure(root,
                                      lidar = 'ouster'
                                      )
            
        self.files,name = kitti_struct._get_point_cloud_file_()
        
        
        #row_label_file = os.path.join(root,dataset,sequence,'point_row_labels.pkl')
        #assert os.path.isfile(row_label_file), "Row label file does not exist " + row_label_file
        #with open(row_label_file, 'rb') as f:
        #    self.row_labels = pickle.load(f)
    

        # Load dataset and laser settings
        print("\n" + "*"*30)
        print("Loading eval dataset...")
        print(f'Number of files: {len(self.files)}')
        print("\n" + "*"*30)
        
        # Load dataset and laser settings
        self.num_samples = len(self.files)
        
        if debug == True:
            self.set_debug()

        n_points = len(self.files)
        self.table = np.zeros((n_points,n_points))
        if self.memory == "RAM":
            self.load_to_RAM()

        self.idx_universe = np.arange(self.num_samples)
   

    def load_to_RAM(self):
        self.memory=="RAM"
        indices = list(range(self.num_samples))
        self.data_on_ram = []
        for idx in tqdm(indices,"Load to RAM"):
            plt = self.modality(self.files[idx],self.augmentation)
            self.data_on_ram.append(plt)

    def set_debug(self):
        indices = np.random.randint(0,self.num_samples,20)
        self.files = self.files[indices]
        self.num_samples = len(indices) # Update number of files

    def __str__(self):
        #name = '-'.join(self.sequence)
        return f'eval-{self.sequence}'
    
    def get_gt_map(self):
        return(self.table)
    
    def __getitem__(self,index):
        
        if self.memory=="RAM":
            pcl = self.data_on_ram[index]
        else:
            pcl = self.modality(self.files[index],self.augmentation)#.long()

        return(pcl,index)

    def __len__(self):
        return(len(self.idx_universe))
        

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)

    def todevice(self,device):
        self.device = device
        
        
    # ==================================================================================================
    def get_anchor_idx(self):
        return []
    
    def get_pose(self):
        return []
    
    def get_row_labels(self):
        return []
    

