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
                                    default = 'GreenHouse',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "e3",type = str)
    parser.add_argument('--plot_path',default  = True ,type = bool)
    parser.add_argument('--record_gif',default  = True ,type = bool)
    parser.add_argument('--rot_angle',default  = 0 ,type = int,
                        help='rotation angle in degrees to rotate the path. the path is rotated at the goemtrical center, ' + 
                        "positive values rotate anti-clockwise, negative values rotate clockwise")
    parser.add_argument('--pose_data_source',default  = "poses" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    
    
    args = parser.parse_args()

    data_dir = "extracted"
    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot_path
    record_gif_flag = args.record_gif
    rotation_angle = args.rot_angle

    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )
    print("[INF] Plotting Flag:   " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    print("[INF] Rotation Angle:  " + str(rotation_angle))

    ground_truth = {'pos_range': 2, # Loop Threshold [m]
                    'num_pos': 1,
                    'warmupitrs': 300, # Number of frames to ignore at the beguinning
                    'roi': 200}
    
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

    save_root_dir  = os.path.join(root_dir,dataset,seq,data_dir)
    if args.debug_mode:
        save_root_dir = os.path.join("temp",dataset,seq,data_dir)
    
    os.makedirs(save_root_dir,exist_ok=True)

    print("[INF] Saving data to directory: %s\n" % save_root_dir)

    from dataloader.kitti.kitti_dataset import load_positions
    from dataloader.utils import gen_ground_truth,rotate_poses
    
    assert args.pose_data_source in ['gps','poses','positions'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    
    poses = load_positions(pose_file)

    print("[INF] Reading poses from: %s"% pose_file)
    print("[INF] Number of poses: %d"% poses.shape[0])

    xy = rotate_poses(poses.copy(),rotation_angle)
    if plotting_flag == True:
        

        fig, ax = plt.subplots()
        ax.scatter(xy[:,0],xy[:,1],s=10,c='black')
        ax.set_aspect('equal')
        plt.show()

    if record_gif_flag:
        anchors,positives = gen_ground_truth(xy,**ground_truth)
        n_points = poses.shape[0]
        # Generate Ground-truth Table 
        # Rows: Anchors
        table = np.zeros((n_points,n_points))
        for anchor,positive in zip(anchors,positives):
            table[anchor,positive] = 1

        print("[INF] Number of points: " + str(n_points))
        gif_file = os.path.join(save_root_dir,'anchor_positive_pair.gif')
        viz_overlap(xy,table,
                    record_gif = True,
                    file_name  = gif_file,
                    frame_jumps= 10)

        print("[INF] Saving gif to: %s"% gif_file)

 


    

