


import os
import numpy as np
from tqdm import tqdm

def load_sync_indices(file):
    overlap = []
    for f in open(file):
        f = f.split(':')[-1]
        indices = [int(i) for i in f.split(' ')]
        overlap.append(indices)
    return(np.array(overlap[0]))

def load_pose_to_RAM(file):
    assert os.path.isfile(file)
    pose_array = []
    for line in tqdm(open(file), 'Loading to RAM'):
        values_str = line.split(' ')
        values = [float(v) for v in values_str]
        pose_array.append(values[0:3])
    return(np.array(pose_array))

def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([f.split('.')[0] for f in os.listdir(target_dir)])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])
    return(files[idx],fullfiles[idx])


def gen_ground_truth(   poses,
                        pos_range= 0.05, # Loop Threshold [m]
                        neg_range= 10,
                        num_neg =  10,
                        num_pos =  10,
                        warmupitrs= 10, # Number of frames to ignore at the beguinning
                        roi       = 5 # Window):
                    ):

    indices = np.array(range(poses.shape[0]-1))
    
    ROI = indices[warmupitrs:]
    anchor   = []
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



def load_dataset(dataset,session,modality,max_points=50000):

    dataset = dataset.lower()
    assert dataset in ['kitti','orchards-uk','pointnetvlad'],'Dataset Name does not exist!'
    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'

    if dataset == 'kitti':
        # Kitti 
        from dataloader.kitti.kitti import KITTI
        
        loader = KITTI( root = session[root_dir],
                        modality = modality,
                        train_loader  = session['train_loader'],
                        val_loader    = session['val_loader'],
                        max_points    = session['max_points']
                        )
    


    elif dataset == 'orchards-uk' :

        from .ORCHARDS import ORCHARDS

        loader = ORCHARDS(root    = session[root_dir],
                            train_loader  = session['train_loader'],
                            test_loader   = session['val_loader'],
                            memory        = memory,
                            )
    
    
    elif dataset == 'pointnetvlad':
        
        from .POINTNETVLAD import POINTNETVLAD
        
        loader = POINTNETVLAD(root       = session[root_dir],
                            train_loader = session['train_loader'],
                            val_loader   = session['val_loader'],
                            memory       = memory
                            )
    

    elif dataset == 'fuberlin':
        
        #session['train_loader']['root'] =  session[root_dir]
        session['val_loader']['root'] =  session[root_dir]
        loader = FUBERLIN(
                            train_loader  = session['train_loader'],
                            val_loader    = session['val_loader'],
                            mode          = memory,
                            max_points = 50000
                            )
    
    return(loader)
