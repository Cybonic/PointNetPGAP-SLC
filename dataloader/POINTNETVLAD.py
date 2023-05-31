
import os
from tqdm import tqdm
import numpy as np
from utils.retrieval import gen_ground_truth, comp_gt_table
import yaml
from torch.utils.data import DataLoader
import torchvision.transforms as Tr
import torch
from .laserscan import LaserData
import pandas as pd
import pickle
import random

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def subsampler(universe,num_sub_samples):
    if not  len(universe)>num_sub_samples:
        num_sub_samples = len(universe)
    return np.random.randint(0,len(universe),size=num_sub_samples)

def load_pc_file(file):
	#returns Nx3 matrix
	pc=np.fromfile(file, dtype=np.float64)

	if(pc.shape[0]!= 4096*3):
		print("Error in pointcloud shape")
		return np.array([])

	pc=np.reshape(pc,(pc.shape[0]//3,3))
	return pc

def load_pose_file(filename):
    if not 'locations' in filename:
        filename = fixe_name(filename)

    with open(filename, 'rb') as handle:
        queries = pd.read_csv(handle).to_numpy()[:,1:]
        print("pose Loaded")
    return queries

def fixe_name(filename):
    file_structure = filename.split(os.sep)
    file_path = os.sep.join(file_structure[:-2])
    file_name = file_structure[-2].split('_')
    new_file_name = os.path.join(file_path,file_name[0] + '_' + 'locations' +'_'+ file_name[1] + '_' + file_name[2] + '.csv')
    return(new_file_name)

def load_pc_files(filenames):
    pcs=[]
    if not isinstance(filenames,list):
        filenames = [filenames]
    for filename in filenames:
		#print(filename)
        pc=load_pc_file(filename)
        if(pc.shape[0]!=4096):
            continue
        pcs.append(pc)
    pcs=np.array(pcs)
    return pcs

def get_queries_dict(filename):
	#key:{'query':file,'positives':[files],'negatives:[files], 'neighbors':[keys]}
	with open(filename, 'rb') as handle:
		queries = pickle.load(handle)
		print("Queries Loaded.")
		return queries

def gather_files(queries:list):
    file_buffer = []
    for k,v in queries.items():
        file_buffer.append(v['query'])
    return file_buffer


def get_query_tuple(root,dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    """
        This function returns two a dict. with the following data 
        # {'pcl':[],'pose':[]}
        # both fields have the same data structure [query,pos,neg,neg2] 

    """
	#get query tuple for dictionary entry
	#return list [query,positives,negatives]
    query_file  = os.path.join(root,dict_value["query"])
    query = load_pc_files(query_file) #Nx3
    query_pose = dict_value['pose']

    # ==========================================================================
    # Get Positive files
    random.shuffle(dict_value["positives"])

    pos_files=[]
    pos_poses=[]
    
    if num_pos > len(dict_value["positives"]):
        num_pos = len(dict_value["positives"])

    for i in range(num_pos):
        indice = dict_value["positives"][i]
        pos_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
        pos_poses.append(QUERY_DICT[indice]["pose"])
    
    positives  = load_pc_files(pos_files)

    # ==========================================================================
    # Get Negatives
    neg_files=[]
    neg_indices=[]
    neg_poses = []

    if num_neg > len(dict_value["negatives"]):
        num_neg = len(dict_value["negatives"])

    if(len(hard_neg)==0):
        random.shuffle(dict_value["negatives"])
        for i in range(num_neg):
            indice = dict_value["negatives"][i]
            neg_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
            neg_poses.append(QUERY_DICT[indice]["pose"])

            ne = dict_value["negatives"][i]
            neg_indices.append(ne)
    else:

        random.shuffle(dict_value["negatives"])
        for i in hard_neg:
            neg_files.append(QUERY_DICT[i]["query"])
            neg_indices.append(i)
        j=0
        while(len(neg_files)<num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                indice = dict_value["negatives"][j]
                neg_files.append(os.path.join(root,QUERY_DICT[indice]["query"]))
                neg_poses.append(QUERY_DICT[indice]["pose"])
                neg_indices.append(dict_value["negatives"][j])
                j+=1

    negatives = load_pc_files(neg_files)

    # ==========================================================================
    # Get HARD Negatives
    if(other_neg==False):
        return [query,positives,negatives]
	#For Quadruplet Loss
    else:
		#get neighbors of negatives and query
        neighbors=[]
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        possible_negs= list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        
        if(len(possible_negs)==0):
            return [query, positives, negatives, np.array([])]
        
        indice = possible_negs[0]
        neg2= load_pc_files(os.path.join(root,QUERY_DICT[indice]["query"]))
        neg2_poses = QUERY_DICT[indice]["pose"]

    # Original implementation does not return Pose
    #return {'pcl':[query,positives,negatives,neg2],'pose':[query_pose,pos_poses,neg_poses,neg2_poses]}
    return {'q':query,'p':positives,'n':negatives,'hn':neg2},{'q':query_pose,'p':pos_poses,'n':neg_poses,'hm':neg2_poses}


def load_picklet(root,filename):
        
    pickle_file = os.path.join(root,filename)
    assert os.path.isfile(pickle_file),'target file does nor exist: ' + pickle_file

    queries = get_queries_dict(pickle_file)
    return queries 


# ===================================================================================================================
#       
#
#
# ===================================================================================================================


class PointNetDataset():
    def __init__(self,
                    root,
                    dataset,
                    pickle_file, # choose between train and test files
                    num_neg, # num of negative samples
                    num_pos, # num of positive samples
                    modality,
                    aug,
                    max_points,
                    memory='RAM', # mode of loading data: [Disk, RAM]
                    **argv):
        
        self.plc_files  = []
        self.plc_names  = []
        self.anchors    = []
        self.positives  = []
        self.negatives  = []

        self.modality = modality
        self.aug = aug
        self.num_neg = num_neg
        self.num_pos = num_pos
        self.hard_neg = 0
        self.memory = memory # mode of loading data: [Disk, RAM]
        #self.ground_truth_mode = argv['ground_truth']

        self.root = root
        # Stuff related to the data organization
        self.base_path = os.path.join(root,dataset,'benchmark_datasets')
        path = os.path.join(root,dataset)
        self.data = load_picklet(path,pickle_file)
        self.pickle_file = pickle_file
        
        self.num_samples  = len(self.data.keys())
        # self.num_samples  = 100
        # Stuff related to sensor parameters for obtaining final representation
        cfg_file = os.path.join('dataloader','pointnetvlad-cfg.yaml')
        sensor_cfg = yaml.safe_load(open(cfg_file , 'r'))
        
        dataset_param = sensor_cfg['oxford'] # Change this 
        sensor =  sensor_cfg[dataset_param['sensor']]

        if modality in ['range']:
            self.param = dataset_param['RP']
        elif modality in ['bev']:
            self.param = dataset_param['BEV']
        else:
            self.param = {}

        self.laser = LaserData(
                parser = None,
                project=True,
                max_points = max_points,
                **dataset_param
                #**argv
                )
        
        # Get anchor, positive and negative indices
        self.anchors, self.positives, self.negatives = self.get_triplet_idx()
        # Compute ground truth table
        self.table = np.zeros((self.num_samples,self.num_samples))
        for a,pp in zip(self.anchors,self.positives):
            for p in pp:
                self.table[a,p]=1

    def __len__(self):
        return(self.num_samples)

    def load_modality(self,file):
        self.laser.open_scan(file)
        pclt = self.laser.get_data(self.modality,self.param)
        return pclt

    def get_triplet_idx(self)-> dict :
        
        positives = []
        negatives = []
        anchors = []
        num_pos = self.num_pos 
        num_neg = self.num_neg
        self.num_samples

        for i in tqdm(range(self.num_samples),"Loading Data", ncols=100):
        
            # ------------------------------------------------------
            # Anchor samples
            # ------------------------------------------------------
            tuple_idx = self.data[i]
            an_pose = np.array(tuple_idx['pose'],dtype=np.float32).reshape(1,-1)
            # ------------------------------------------------------
            # Positives samples
            # ------------------------------------------------------
            pos = tuple_idx['positives']
            if len(pos)==0:
                continue
            
            anchors.append(i)
            # get positives' poses
            pos_pose = np.array([np.array(self.data[ii]['pose'],dtype=np.float32) for ii in pos])
            # Compute distance wrt to anchor
            dist = np.sqrt(np.sum((an_pose -pos_pose)**2,axis=1))
            # print(dist[np.argsort(dist)])
            select_pos = np.array(pos)[np.argsort(dist)]

            if num_pos > len(select_pos):
                num_pos = len(select_pos)
                
            # Generate the positive indices
            idx = np.arange(0,num_pos,dtype=np.int32)
            pos_idx_vec = np.array(select_pos)[idx]
            positives.append(pos_idx_vec)
            # ------------------------------------------------------
            # Negative samples
            # ------------------------------------------------------
            if num_neg == None:
                continue
            select_neg = tuple_idx['negatives']
            random.shuffle(negatives) # Shuffle the negative indices
            # set number of negatives to retrieve equal the actual size of negatives, 
            # when not enough negatives exist
            if num_neg > len(negatives):
                num_neg = len(negatives)
            idx = np.arange(0,self.num_neg,dtype=np.int32)
            neg_idx_vec = np.array(select_neg)[idx]
            negatives.append(neg_idx_vec)

        return anchors,positives,negatives	


    

class PointNetTriplet(PointNetDataset):
    def __init__(self,
                root,
                dataset,
                sequence, # choose between train and test files
                num_neg   = 18, # num of negative samples
                num_pos   = 1, # num of positive samples
                modality  = 'range',
                aug       = False,
                memory    = 'RAM',
                max_points = 10000,
                **argv
                ):

        super(PointNetTriplet,self).__init__(root,dataset, sequence, num_neg, num_pos, 
                                            modality, aug, max_points, **argv)
        self.modality = modality
        self.memory   = memory
        self.preprocessing = PREPROCESSING

        # COPY IDX FROM DATASET LOADER
        self.an_train  = self.anchors
        self.pos_train = self.positives 
        self.neg_train = self.negatives
        self.table_train = self.table
        self.device = 'cpu'

        self.num_samples = len(self.anchors)
        # TRIPLET DATA
        self.an_struct, self.pos_struct, self.neg_struct = self.load_triplet_data(self.an_train,self.pos_train,self.neg_train)
        
    def load_triplet_data(self,anchors,positives,negatives):
        #file_buffer = gather_files(self.queries)
        neg = {'files':[],'poses':[]}
        pos = {'files':[],'poses':[]}
        an  = {'files':[],'poses':[]}

        for i in tqdm(range(self.num_samples),"Loading to RAM"):
            
            anchor = anchors[i]
            file = self.data[anchor]
            files = os.path.join(self.base_path,file['query']) # build the pcl file path
            poses = np.array(self.data[anchor]['pose'],dtype=np.float32).reshape(1,-1)
            an['files'].append(files)
            an['poses'].append(poses)
            
            pos_idx = positives[i]
            files = [os.path.join(self.base_path,self.data[j]['query']) for j in pos_idx]
            poses =[np.array(self.data[j]['pose'],dtype=np.float32) for j in pos_idx]
            pos['files'].append(files)
            pos['poses'].append(poses)

            neg_idx = negatives[i]
            files = [os.path.join(self.base_path,self.data[j]['query']) for j in neg_idx]
            poses = [np.array(self.data[j]['pose'],dtype=np.float32) for j in neg_idx]
            neg['files'].append(files)
            neg['poses'].append(poses)
            
        return(an,pos,neg)
    
    def __getitem__(self,index:int):
        data= self.load_data(index)
        pose= self.load_pose(index)
        return(data,pose)

    def __len__(self):
        return(self.num_samples)

    def load_data(self,idx):
        an_file = self.an_struct['files'][idx]
        self.laser.open_scan(an_file)
        anchor_data = self.preprocessing(self.laser.get_data(self.modality,self.param).round(decimals=6)).to(self.device)
        
        pos_data = []
        for file in self.pos_struct['files'][idx]:
            self.laser.open_scan(file)
            pos_data.append(self.preprocessing(self.laser.get_data(self.modality,self.param)).round(decimals=6).to(self.device))
        pos_data = torch.stack(pos_data,axis=0)
        
        neg_data = []
        for file in self.neg_struct['files'][idx]:
            self.laser.open_scan(file)
            neg_data.append(self.preprocessing(self.laser.get_data(self.modality,self.param)).round(decimals=6).to(self.device))
        neg_data = torch.stack(neg_data,axis=0)

        return {'anchor':anchor_data,'positive':pos_data,'negative':neg_data}


    def load_pose(self,idx):
        an_pose = self.an_struct['poses'][idx]
        an_pose = torch.tensor(an_pose)
        
        pos_data = []
        for pose in self.pos_struct['poses'][idx]:
            pos_data.append(pose)
        pos_data = torch.tensor(pos_data)
        
        neg_data = []
        for pose in self.neg_struct['poses'][idx]:
            neg_data.append(pose)
        neg_data = torch.tensor(neg_data)
        return {'anchor':an_pose,'positive':pos_data,'negative':neg_data}


    def get_idx_universe(self):
        return self.idx_universe
    
    def get_anchor_idx(self):
        return(self.an_train)
    
    def get_gt_map(self):
        return(self.table_train)
    
    def get_pose(self):
        return(0)

    def __str__(self):
        seq_name = self.pickle_file.split('.')[0]
        return f'pointnetvlad-{seq_name}'

    def todevice(self,device):
        self.device = device


# ===============================================================================
#
#
#
# ===============================================================================

class PointNetEval(PointNetDataset):
    def __init__(self,
                root,
                dataset,
                sequence, # choose between train and test files
                num_pos   = 1, # num of positive samples
                modality  = 'range',
                memory    = 'RAM',
                max_points = 10000,
                **argv
                ):
        
        super(PointNetEval,self).__init__(root,dataset, sequence, None , num_pos, 
                                            modality, None , max_points, **argv)
        self.modality = modality
        #self.idx_universe = np.arange(50)
        self.idx_universe = np.arange(self.num_samples)
        self.preprocessing = PREPROCESSING

        # COPY IDX FROM DATASET LOADER
        self.an_eval  = self.anchors
        self.pos_eval = self.positives 
        self.neg_eval = self.negatives
        self.table_eval = self.table
        self.device = 'cpu'

    def todevice(self,device):
        self.device = device

    def __len__(self):
        return(len(self.idx_universe))
    
    def __getitem__(self,idx):
        data,pose,positives_idx = self.load_data(idx)
        data = self.preprocessing(data).to(self.device)
        return data,idx

    def load_data(self,idx):
        idx = self.idx_universe[idx]
        file = self.data[idx]['query']
        full_file_path = os.path.join(self.base_path,file) # build the pcl file path
        data = self.load_modality(full_file_path)
        pose = np.array(self.data[idx]['pose'],dtype=np.float32).reshape(1,-1)
        positives_idx = self.data[idx]['positives']
        return data,pose,positives_idx
    
    def _get_modality_(self,idx):
        idx = self.idx_universe[idx]
        file = self.data[idx]['query']
        full_file_path = os.path.join(self.base_path,file) # build the pcl file path
        data = self.load_modality(full_file_path)
        return data
    
    def __call__(self,idx):
        return self._get_modality_(idx)
    
    # OUTPUT CALLS
    def get_gt_map(self):
        return self.table_eval
    
    def get_idx_universe(self):
        return self.idx_universe
    
    def get_anchor_idx(self):
        return self.an_eval

    def __str__(self):
        seq_name = self.pickle_file.split('.')[0]
        return f'pointnetvlad-{seq_name}'



class POINTNETVLAD():
    def __init__(self,**argv):
        
        self.root = argv['root']
        self.train_loader = argv['train_loader']
        self.val_loader = argv['val_loader']
        self.memory = argv['memory']
  
    def get_train_loader(self):
        # TRAINING DATALOADER
        train = PointNetTriplet(self.root,
                    **self.train_loader['data'],
                    memory = self.memory
                    )

        self.trainloader = DataLoader(train,
                    batch_size = self.train_loader['batch_size'],
                    num_workers= 0,
                    pin_memory=False,
                    )
        
        return self.trainloader
    
    def get_val_loader(self):
        # EVALUATION DATALOADER
        test = PointNetEval(self.root,
                    **self.val_loader['data'],
                    memory = self.memory
                    )

        self.testloader = DataLoader(test,
                    batch_size = self.val_loader['batch_size'],
                    num_workers= 0,
                    pin_memory=False,
                    )
        
        return self.testloader

