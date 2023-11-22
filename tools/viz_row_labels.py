import os, sys
import numpy as np
import argparse

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


from utils.viz import myplot
from dataloader.utils import extract_points_in_rectangle_roi
import yaml

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

from tqdm import tqdm
import pickle

def viz_overlap(xy, loops, record_gif= False, file_name = 'anchor_positive_pair.gif',frame_jumps=50):

    

    mplot = myplot(delay=0.2)
    mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'whitesmoke')
    mplot.xlabel('m'), mplot.ylabel('m')
        
    if record_gif == True:
        mplot.record_gif(file_name)
    
    indices = np.array(range(xy.shape[0]-1))
    ROI = indices[2:]
    positives  = []
    anchors    = []
    pos_distro = []
    n_points   = ROI.shape[0]

    color = np.array(['y' for ii in range(0,n_points)])
    scale = np.array([10 for ii in range(0,n_points)])
    
    for i in tqdm(ROI,desc='Vizualizing Loop Closure'):
        idx = loops[i]        
        
        positives_idx = np.where(idx>0)[0]
  
        if len(positives_idx)>0:
            positives.extend(positives_idx)
            anchors.append (i)
            pos_distro.append(positives_idx)
        
        if i % frame_jumps != 0:
            continue

        color[:i]= 'k'
        scale[:i]= 10
        # Colorize the N smallest samples
        color[anchors]   = 'b'
        color[positives] = 'g'

        scale[positives] = 100

        x = xy[:i,1]
        y = xy[:i,0]

        mplot.update_plot(y,x,offset=2,color=color,zoom=0,scale=scale) 

    return(pos_distro)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'GreenHouse',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "e3",type = str)
    parser.add_argument('--show',default  = True ,type = bool)
    parser.add_argument('--pose_data_source',default  = "positions" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    parser.add_argument('--save_data',default  = True ,type = bool,
                        help='save evaluation data to a pickle file')
    
    
    args = parser.parse_args()

    data_dir = "extracted"
    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    show = args.show

    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )
    print("[INF] Plotting Flag:   " + str(show))
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    print("[INF] Root directory: %s\n"% root_dir)

    dir_path = os.path.join(root_dir,dataset,seq,data_dir)
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path
    
    print("[INF] Loading data from directory: %s\n" % dir_path)

    # Create Save Directory
    save_root_dir  = os.path.join(root_dir,dataset,seq,"extracted")
    if args.debug_mode:
        save_root_dir = os.path.join('temp',dataset,seq)
    
    os.makedirs(save_root_dir,exist_ok=True)
    print("[INF] Saving data to directory: %s\n" % save_root_dir)

    # Loading DATA
    from dataloader.kitti.kitti_dataset import load_positions
    from dataloader.utils import rotate_poses
    from dataloader import row_segments
    
    assert args.pose_data_source in ['gps','poses','positions'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    poses = load_positions(pose_file)

    print("[INF] Reading poses from: %s"% pose_file)
    print("[INF] Number of poses: %d"% poses.shape[0])

    # Load row ids
    seq = seq.replace('/','_')
    print("[INF] Loading row segments from: %s"% seq)
    dataset_raws = row_segments.__dict__[seq]
    # Load aline rotation
    rotation_angle = dataset_raws['angle']
    # Rotate poses to match the image frame
    rotated_poses = rotate_poses(poses.copy(),rotation_angle)
    xy = rotated_poses[:,:2]

    # Load row segments
    rectangle_rois = np.array(dataset_raws['rows'])
    labels  = extract_points_in_rectangle_roi(xy,rectangle_rois)

    unique_labels = np.unique(labels)

    n_points = xy.shape[0]

    # Generate Ground-truth Table
    if args.save_data:
    
        file = os.path.join(save_root_dir,'point_row_labels.pkl')

         # save the numpy arrays to a file using pickle
        with open(file, 'wb') as f:
            pickle.dump(labels, f)
    
        print("[INF] saved ground truth at:" + file)


   
    
    if show:
        print("[INF] Plotting data...")
         # color pallet based on the number of unique labels
        color_pallet = np.array(['y','b','g','r','c','m','k','w'])
        color = color_pallet[labels]

        point_color = np.array([color_pallet[labels[ii]] for ii in range(0,n_points)])
        # Plot the data
        fig = plt.figure()
        fig, ax = plt.subplots()
        ax.scatter(xy[:,0],xy[:,1],s=10,c=point_color)
        ax.set_aspect('equal')
        plt.savefig(os.path.join(save_root_dir,'point_row_labels.png'))
        plt.show()

 


    
