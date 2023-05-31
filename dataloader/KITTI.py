
import os
from tqdm import tqdm
import torchvision.transforms as Tr
import torch

import numpy as np
import yaml
from torch.utils.data import DataLoader
from .laserscan import LaserData
from .utils import get_files

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def load_pose_to_RAM(file):
    assert os.path.isfile(file)
    pose_array = []
    for line in tqdm(open(file), 'Loading to RAM'):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        position = values[[3,7,11]]
        #position[:,1:] =position[:,[2,1]] 
        pose_array.append(position.tolist())

    pose_array = np.array(pose_array)   
    pose_array[:,1:] =pose_array[:,[2,1]] 
    return(pose_array)

def subsampler(universe,num_sub_samples):
    if not  len(universe)>num_sub_samples:
        num_sub_samples = len(universe)
    return np.random.randint(0,len(universe),size=num_sub_samples)

def parse_triplet_file(file):
    assert os.path.isfile(file)
    f = open(file)
    anchors = []
    positives = []
    negatives = []
    for line in f:
        value_str = line.rstrip().split('_')
        anchors.append(int(value_str[0].split(':')[-1]))
        positives.append(int(value_str[1].split(':')[-1]))
        negatives.append([int(i) for i in value_str[2].split(':')[-1].split(' ')])
    f.close()

    anchors = np.array(anchors,dtype=np.uint32)
    positives = np.array(positives,dtype=np.uint32)
    negatives = np.array(negatives,dtype=np.uint32)

    return anchors,positives,negatives

def gen_ground_truth(   poses,
                        pos_range= 0.05, # Loop Threshold [m]
                        neg_range=10,
                        num_neg = 10,
                        num_pos = 10,
                        warmupitrs= 10, # Number of frames to ignore at the beguinning
                        roi       = 5 # Window):
                    ):

    indices = np.array(range(poses.shape[0]-1))
    
    ROI = indices[warmupitrs:]
    anchor =   []
    positive = []

    for i in ROI:
        
        _map_   = poses[:i,:]
        pose    = poses[i,:].reshape((1,-1))
        dist_meter  = np.sqrt(np.sum((pose -_map_)**2,axis=1))

        pos_idx = np.where(dist_meter[:i-roi] < pos_range)[0]
        
        if len(pos_idx)>0:
            min_sort = np.argsort(dist_meter[pos_idx])
            if num_pos == -1:
                pos_select = pos_idx[min_sort]
            else:
                pos_select = pos_idx[min_sort[:num_pos]]

            positive.append(pos_select)
            anchor.append(i)
    
    # Negatives
    negatives= []
    neg_idx = np.arange(num_neg)   
    for a, pos in zip(anchor,positive):
        pa = poses[a,:].reshape((1,-1))
        dist_meter = np.sqrt(np.sum((pa-poses)**2,axis=1))
        neg_idx = np.where(dist_meter > neg_range)[0]
        neg_idx = np.setxor1d(neg_idx,pos)
        select_neg = np.random.randint(0,len(neg_idx),num_neg)
        neg_idx = neg_idx[select_neg]
        negatives.append(neg_idx)

    return(anchor,positive,negatives)


class kitti_velo_parser():
    def __init__(self):
        self.dt = []

    def velo_read(self,scan_path):
        scan = np.fromfile(scan_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        return(np.array(scan))

# ===================================================================================================================
#       
#
#
# ===================================================================================================================
class FileStruct():
    def __init__(self,root,dataset,sequence,sync = True):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        #for seq in sequences:
        self.target_dir = os.path.join(root,dataset,sequence)
        #self.target_dir.append(target_dir)
        assert os.path.isdir(self.target_dir),'target dataset does nor exist: ' + self.target_dir

        pose_file = os.path.join(self.target_dir,'poses.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_pose_to_RAM(pose_file)
        #self.pose.extend(pose)

        point_cloud_dir = os.path.join(self.target_dir,'velodyne')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        self.file_names, self.point_cloud_files = get_files(point_cloud_dir)

    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files,self.file_names)
        return(self.point_cloud_files[idx],self.file_names[idx])
    
    def _get_pose_(self):
        return(self.pose)

    
    def _get_target_dir(self):
        return(self.target_dir)


class KittiDataset():
    def __init__(self,
                    root,
                    dataset,
                    seq,
                    modality = 'pcl' ,
                    ground_truth = { 'pos_range':4, # Loop Threshold [m]
                                     'neg_range': 10,
                                     'num_neg':20,
                                     'num_pos':1,
                                     'warmupitrs': 600, # Number of frames to ignore at the beguinning
                                     'roi':500},
                        **argv):
        
        self.plc_files  = []
        self.plc_names  = []
        self.poses      = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []
        self.modality = modality
        baseline_idx  = 0 
        #self.ground_truth_mode = argv['ground_truth']
        assert isinstance(seq,list)

        for seq in seq:
            kitti_struct = FileStruct(root, dataset, seq)
            files,name = kitti_struct._get_point_cloud_file_()
            self.plc_files.extend(files)
            self.plc_names.extend(name)
            pose = kitti_struct._get_pose_()
            self.poses.extend(pose)
            target_dir = kitti_struct._get_target_dir()
            # Load indicies to split the dataset in queries, positive and map 
            anchors,positives,negatives = gen_ground_truth(pose,**ground_truth)
            
            self.anchors.extend(baseline_idx + np.array(anchors))
            self.positives.extend(baseline_idx + np.array(positives))
            self.negatives.extend(baseline_idx + np.array(negatives))

            baseline_idx += len(files)

        # Load dataset and laser settings
        self.poses = np.array(self.poses)
        cfg_file = os.path.join('dataloader','kitti-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        
        self.num_samples = len(self.plc_files)
        dataset_param = sensor_cfg[seq]
        sensor =  sensor_cfg[dataset_param['sensor']]

        if not 'square_roi' in argv:
            argv['square_roi'] = [sensor_cfg[seq]['roi']]

        if modality in ['range']:
            self.param = sensor_cfg[seq]['RP']
        elif modality in ['bev']:
            self.param = sensor_cfg[seq]['BEV']
        else:
            self.param = {}

        self.laser = LaserData(
                parser = kitti_velo_parser(),
                project=True,
                **argv
                )

        n_points = baseline_idx
        self.table = np.zeros((n_points,n_points))
        for a,pos in zip(self.anchors,self.positives):
            for p in pos:
                self.table[a,p]=1
    
    def __len__(self):
        return(self.num_samples)

    def __call__(self,idx):
        return self.load_point_cloud(idx)

    def _get_gt_(self):
        return self.table

    def _get_pose_(self):
        return(self.poses)

    def load_point_cloud(self,idx):
        file = self.plc_files[idx]
        self.laser.open_scan(file)
        pclt = self.laser.get_pcl()
        return pclt
        
    def _get_modality_(self,idx):
        file = self.plc_files[idx]
        self.laser.open_scan(file)
        pclt = self.laser.get_data(self.modality,self.param)
        return pclt




class KITTIEval(KittiDataset):
    def __init__(self,root, dataset, sequence, sync = True,   # Projection param and sensor
                modality = 'range' , 
                mode = 'Disk', 
                **argv
                ):
        
        super(KITTIEval,self).__init__(root, dataset, sequence, sync=sync, modality=modality,**argv)
        self.modality = modality
        self.mode     = mode
        self.preprocessing = PREPROCESSING
        self.num_samples = self.num_samples
        self.sequence = sequence
        
        self.device='cpu'

        self.idx_universe = np.arange(self.num_samples)
        #self.idx_universe = np.arange(50)
        self.map_idx  = np.setxor1d(self.idx_universe,self.anchors)
        self.poses = self._get_pose_()
        #assert len(np.intersect1d(self.anchors,self.map_idx)) == 0, 'No indicies should be in both anchors and map'
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()
        
    def __str__(self):
        seq_name = '_'.join(self.sequence)
        return f'kitti-{seq_name}'
    
    def get_gt_map(self):
        return(self.table)
    
    def get_eval_data(self,index):
        global_index = self.idx_universe[index] # Only useful when subsampler is on
        
        if self.mode == 'RAM':
            data = self.inputs[global_index]         
        elif self.mode == 'Disk':
            data = self._get_modality_(global_index)
        
        return(data,global_index)

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_modality_(i)
            img[i]=data#.astype(np.uint8)
        return img

    def __call__(self,idx):
        pcl,gindex = self.get_eval_data(idx)
        return pcl
    
    def __getitem__(self,index):
        pcl,gindex = self.get_eval_data(index)
        pcl = self.preprocessing(pcl).to(self.device)
        return(pcl,gindex)

    def __len__(self):
        return(len(self.idx_universe))
        
    def get_pose(self):
        pose = self._get_pose_()
        return pose

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)

    def todevice(self,device):
        self.device = device


class KITTITriplet(KittiDataset):
    def __init__(self,
                    root,
                    dataset,
                    sequence, 
                    memory='Disk', 
                    modality = 'projection', 
                    aug=False,
                    **argv):
        super(KITTITriplet,self).__init__(root,dataset,sequence, modality=modality,aug=aug,**argv)

        self.modality = modality
        self.aug_flag = aug
        self.mode     = memory
        self.preprocessing = PREPROCESSING
        self.eval_mode = False
        self.idx_universe = np.arange(self.num_samples) # 
        self.device='cpu'
        # Load to RAM
        if self.mode == 'RAM':
            self.inputs = self.load_RAM()

        #if 'subsample' in argv:
        self.anchors = subsampler(self.anchors,int(len(self.anchors)/10))
        #pass

    def load_RAM(self):
        img   = {}       
        for i in tqdm(self.idx_universe,"Loading to RAM"):
            data  = self._get_modality_(i)
            img[i]=data#.astype(np.uint8)
        return img


    def get_triplet_data(self,index):
        
        an_idx,pos_idx,neg_idx  = self.anchors[index],self.positives[index], self.negatives[index]
        if self.mode == 'RAM':     
            # point clouds are already converted to the input representation, is only required to 
            #  convert to tensor 
            plt_anchor = self.preprocessing(self.inputs[an_idx]).to(self.device)
            plt_pos = torch.stack([self.preprocessing(self.inputs[i]).to(self.device) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self.inputs[i]).to(self.device) for i in neg_idx],axis=0)

        elif self.mode == 'Disk':
            plt_anchor = self.preprocessing(self._get_modality_(an_idx)).to(self.device)
            plt_pos = torch.stack([self.preprocessing(self._get_modality_(i)).to(self.device) for i in pos_idx],axis=0)
            plt_neg = torch.stack([self.preprocessing(self._get_modality_(i)).to(self.device) for i in neg_idx],axis=0)
        else:
            raise NameError

        pcl = {'anchor':plt_anchor,'positive':plt_pos,'negative':plt_neg}
        indx = {'anchor':len(plt_anchor),'positive':len(plt_pos),'negative':len(plt_neg)}
        return(pcl,indx)

    def get_eval_data(self,index):
        global_index = self.idx_universe[index] # Only useful when subsampler is on
        
        if self.mode == 'RAM':
            data = self.inputs[global_index]         
        elif self.mode == 'Disk':
            data = self._get_modality_(global_index)
        
        plc = self.preprocessing(data)
        return(plc,global_index)
    

    def set_eval_mode(self,mode=True):
        self.eval_mode = mode

    def __getitem__(self,index):
        if not self.eval_mode:
            pcl,indx = self.get_triplet_data(index)
        else:
            pcl,indx = self.get_eval_data(index)
        return(pcl,indx)
    
    def __len__(self):
        if not self.eval_mode:
            num_sample = len(self.anchors)
        else:
            num_sample = len(self.idx_universe)
        return(num_sample)
    
    def get_pose(self):
        return self._get_pose_()

    def get_anchor_idx(self):
        return np.array(self.anchors,np.uint32)

    def get_map_idx(self):
        return np.array(self.map_idx,np.uint32)
    
    def get_idx_universe(self):
        return(self.idx_universe)
    
    def get_gt_map(self):
        return(self.table)
    
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