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

def viz_triplet(xy, loops,show_negatives=False, record_gif= False, file_name = 'anchor_positive_pair.gif',frame_jumps=50):
    """
    Visualize the loop closure process
    Args:
        xy (np.array): Array of poses
        loops (np.array): Array of loop closure indices
        show_negatives (bool, optional): [description]. Defaults to False.
        record_gif (bool, optional): [description]. Defaults to False.
        file_name (str, optional): [description]. Defaults to 'anchor_positive_pair.gif'.
        frame_jumps (int, optional): [description]. Defaults to 50.
    """
    

    mplot = myplot(delay=0.2)
    mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'white')
    mplot.xlabel('m'), mplot.ylabel('m')
        
    if record_gif == True:
        mplot.record_gif(file_name)
    
    indices = np.array(range(xy.shape[0]-1))
    n_points   = xy.shape[0]

    for i in tqdm(indices,desc='Vizualizing Loop Closure'):
        idx = loops[i]        
        
        positives_idx = np.where(idx>0)[0]
        negatives_idx = np.where(idx<0)[0]
        
        if i % frame_jumps != 0 or positives_idx.shape[0] == 0 :
            continue
            
        # Colorize all points based on their index
        color = np.array(['w' for ii in range(0,n_points)])
        scale = np.array([10 for ii in range(0,n_points)])

        color[:i]= 'k'
        scale[:i]= 10

        # Colorize anchors
        color[i]   = 'b'
        scale[i]   = 30

        # Colorize positives
        color[positives_idx] = 'g'
        scale[positives_idx] = 100

        # Colorize negatives
        if show_negatives:
            color[negatives_idx] = 'r'
            scale[negatives_idx] = 100

        x = xy[:i+1,1]
        y = xy[:i+1,0]

        mplot.update_plot(y,x,offset=2,color=color,zoom=0,scale=scale) 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'uk',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    parser.add_argument('--seq',default  = "orchards/june23",type = str)
    parser.add_argument('--plot',default  = True ,type = bool)
    parser.add_argument('--loop_thresh',default  = 1 ,type = float)
    parser.add_argument('--record_gif',default  = False ,type = bool)
    parser.add_argument('--show_negatives',default  = True ,type = bool)
    parser.add_argument('--pose_file',default  = 'poses' ,type = str)
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot
    record_gif_flag = args.record_gif
    loop_thresh = args.loop_thresh

    print("[INF] Dataset Name:  " + dataset)
    print("[INF] Sequence Name: " + str(seq) )
    print("[INF] Plotting Flag: " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    print("[INF] Reading poses from : " + args.pose_file)

    ground_truth = {'pos_range':2, # Loop Threshold [m]
                    'neg_range':15,
                    'num_neg':10,
                    'num_pos':1,
                    'warmupitrs': 200, # Number of frames to ignore at the beguinning
                    'roi':100}
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    pose_file = os.path.join(root_dir,dataset,seq,'extracted','poses.txt')
    print("[INF] Reading poses from : " + pose_file)


    seq = seq.replace('/','_')
    from dataloader.kitti.kitti_dataset import load_pose_to_RAM
    from dataloader import row_ids
    
    # Load raw poses
    poses = load_pose_to_RAM(pose_file)
    # Load row ids
    dataset_raws = row_ids.__dict__[seq]
    # Load aline rotation
    rotation_angle = dataset_raws['angle']
    # Rotate poses to match the image frame
    from dataloader.utils import rotate_poses,gen_gt_constrained_by_rows
    

    xy = rotate_poses(poses.copy(),rotation_angle)
    retangle_rois = np.array(dataset_raws['rows'])
    
    anchors,positives,negatives  = gen_gt_constrained_by_rows(xy,retangle_rois,**ground_truth)

    n_points = xy.shape[0]

    
    # Generate Ground-truth Table 
    # Rows: Anchors
    table = np.zeros((n_points,n_points))
    for anchor,positive,negative in zip(anchors,positives,negatives):
        table[anchor,positive] = 1
        table[anchor,negative] = -1
        

    print("[INF] Number of points: " + str(n_points))

    viz_triplet(xy,table,
                show_negatives=args.show_negatives,
                record_gif=True,
                file_name='triplet.gif',
                frame_jumps=50)

 


    

