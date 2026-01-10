
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

from dataloader.hortov2.dataset import file_structure

import pickle

PREPROCESSING = Tr.Compose([Tr.ToTensor()])


class Eval:
    def __init__(self,  root, 
                        sequence,
                        modality = None ,
                        memory= "DISK", 
                        debug = False,
                        device='cpu',
                        verbose=False
                        ):
        
        assert memory in ["RAM", "DISK"]
        self.memory   = memory 
        self.modality = modality
        self.sequence = sequence
        
        #self.num_samples = self.num_samples
        self.device   = device
        self.struct = file_structure(root,sequence)

        self.files = self.struct._get_point_cloud_files_()
        # nnearest = self.struct._get_nnearest_()

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
            plt = self.modality(self.files[idx],False)
            self.data_on_ram.append(plt)

    def set_debug(self):
        indices = np.random.randint(0,self.num_samples,20)
        self.files = self.files[indices]
        self.num_samples = len(indices) # Update number of files

    def __str__(self):
        #name = '-'.join(self.sequence)
        return f'eval-{self.sequence}'
    
    def get_gt_map(self):
        return NotImplementedError
    
    def __getitem__(self,index):

        pcd = self.struct._load_pcd_(index)
        position = self.struct._get_position_(index)
        label = self.struct._get_label_(index)
        
        if self.memory=="RAM":
            pcl = self.data_on_ram[index]
        else:
            pcl = self.modality(pcd,False)

        # Handle both dense tensors and sparse tensors
        #if hasattr(pcl, 'shape'):
            # Dense tensor (numpy array or torch tensor)
        #    if pcl.shape[-1] > 3:
        #        pcl = pcl[:,:3]
        #elif hasattr(pcl, 'F'):
            # Sparse tensor (e.g., MinkowskiEngine SparseTensor)
            # pcl.F contains the features
        #    if pcl.F.shape[-1] > 3:
        #        pcl.F = pcl.F[:,:3]
        
        return(pcl, index)

    def __len__(self):
        return(len(self.idx_universe))
        

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)

    def todevice(self,device):
        self.device = device

    def get_positions(self):
        return self.struct._get_positions_()

    def get_labels(self):
        return self.struct._get_labels()
    # ==================================================================================================
    def get_anchor_idx(self):
        return []
    
    def get_pose(self):
        return []
    
    def get_row_labels(self):
        return []
    
    def get_ground_truth_loop_closure(self,
                                      warm_up=100,
                                      lower_bound_idx=50,
                                      distance_threshold=2.0,
                                      top_k=1):
        """   """
                  
        return self.struct.get_ground_truth_loop_closure(
                            warm_up=warm_up,
                            lower_bound_idx=lower_bound_idx,
                            distance_threshold=distance_threshold,
                            topk=top_k  # Use all neighbors within threshold
                )
    

    def load_ground_truth(self):
        return self.struct._load_ground_truth()