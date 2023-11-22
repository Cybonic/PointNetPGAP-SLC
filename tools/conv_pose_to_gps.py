import os, sys
import numpy as np
import argparse

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


from utils.viz import myplot
import yaml

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from tqdm import tqdm
from dataloader.kitti.kitti_utils import save_positions_KITTI_format

def conv_to_positions(poses):
    """
    Convert poses to positions coordinates
    
    Parameters
    ----------
    poses : np.array
        Array of poses nx[4x4]
    
    Returns
    -------
    gps : np.array
        Array of position nx3
    """
    assert isinstance(poses,np.ndarray), "Poses must be a numpy array"
    assert poses.shape[1] >= 3, "Poses must be a nx4x4 array"
    # Convert poses to GPS coordinates
    gps = poses[:,:3]
    return gps


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'uk',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "orchards/sum22/extracted",type = str)
    parser.add_argument('--pose_data_source',default  = "gps" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    
    
    args = parser.parse_args()
    root    = args.root
    dataset = args.dataset 
    seq     = args.seq


    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )

    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    print("[INF] Root directory: %s\n"% root_dir)

    dir_path = os.path.join(root_dir,dataset,seq)
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path
    
    print("[INF] Loading data from directory: %s\n" % dir_path)

    save_root_dir  = dir_path
    if args.debug_mode:
        save_root_dir = os.path.join("temp",dataset,seq)
        os.makedirs(save_root_dir,exist_ok=True)

    print("[INF] Saving data to directory: %s\n" % save_root_dir)

    from dataloader.kitti.kitti_dataset import load_positions
    
    assert args.pose_data_source in ['gps','poses'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    
    poses = load_positions(pose_file)

    positions = conv_poses_to_positions(poses)

    file = os.path.join(save_root_dir,'positions.txt')

    save_positions_KITTI_format(save_root_dir,positions)

 


    

