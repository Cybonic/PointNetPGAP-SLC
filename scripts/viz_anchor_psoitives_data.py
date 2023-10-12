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
                                    default = 'uk',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    parser.add_argument('--seq',default  = "orchards/sum22/extracted",type = str)
    parser.add_argument('--plot_path',default  = True ,type = bool)
    parser.add_argument('--loop_thresh',default  = 1 ,type = float)
    parser.add_argument('--record_gif',default  = False ,type = bool)
    parser.add_argument('--pose_file',default  = 'poses' ,type = str)
    parser.add_argument('--rot_angle',default  = -3 ,type = int)
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot_path
    record_gif_flag = args.record_gif
    loop_thresh = args.loop_thresh
    rotation_angle = args.rot_angle

    print("[INF] Dataset Name:  " + dataset)
    print("[INF] Sequence Name: " + str(seq) )
    print("[INF] Plotting Flag: " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    print("[INF] Reading poses from : " + args.pose_file)
    print("[INF] Rotation Angle: " + str(rotation_angle))

    ground_truth = {'pos_range':2, # Loop Threshold [m]
                    'num_pos':1,
                    'warmupitrs': 500, # Number of frames to ignore at the beguinning
                    'roi':400}
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    pose_file = os.path.join(root_dir,dataset,seq,'poses.txt')
    print("[INF] Reading poses from : " + pose_file)


    from dataloader.kitti.kitti_dataset import load_pose_to_RAM
    from dataloader.utils import gen_ground_truth,rotate_poses
    
    poses = load_pose_to_RAM(pose_file)

    if plotting_flag == True:
        xy = rotate_poses(poses.copy(),rotation_angle)

        fig, ax = plt.subplots()
        ax.scatter(xy[:,0],xy[:,1],s=10,c='black')
        ax.set_aspect('equal')
        plt.show()

    if record_gif_flag:

        anchors,positives = gen_ground_truth(poses,**ground_truth)
        n_points = poses.shape[0]
       
        # Generate Ground-truth Table 
        # Rows: Anchors
        table = np.zeros((n_points,n_points))
        for anchor,positive in zip(anchors,positives):
            table[anchor,positive] = 1

        print("[INF] Number of points: " + str(n_points))
        gif_file = os.path.join(root_dir,dataset,seq,'anchor_positive_pair.gif')
        viz_overlap(poses,table,
                    record_gif=True,
                    file_name=gif_file,
                    frame_jumps=100)

 


    

