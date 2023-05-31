


import os
import numpy as np
from tqdm import tqdm
import dataloader


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


def load_dataset(dataset,session,memory,max_points=50000,debug=False):

    dataset = dataset.lower()
    assert dataset in ['kitti','orchards-uk','pointnetvlad'],'Dataset Name does not exist!'
    if os.sep == '\\':
        root_dir = 'root_ws'
    else:
        root_dir = 'root'

    if dataset == 'kitti':
        # Kitti 
        from .KITTI import KITTI
        loader = KITTI( root = session[root_dir],
                        train_loader  = session['train_loader'],
                        val_loader    = session['val_loader'],
                        memory          = memory
                        )
        
    elif dataset == 'orchards-uk' :

        from .ORCHARDS import ORCHARDS

        loader = ORCHARDS(root    = session[root_dir],
                            train_loader  = session['train_loader'],
                            test_loader    = session['val_loader'],
                            memory          = memory,
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
