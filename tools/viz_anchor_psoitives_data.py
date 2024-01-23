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
import pickle
from tools.viz_row_labels import load_row_bboxs

def viz_overlap(xy, loops, record_gif= False, file_name = 'anchor_positive_pair.gif',frame_jumps=50):

    

    mplot = myplot(delay=0.2)
    mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'k')
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
    parser.add_argument('--root', type=str, default='/home/deep/Dropbox/SHARE/DATASET')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'GreenHouse',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "e3/extracted",type = str)
    parser.add_argument('--plot_path',default  = True ,type = bool)
    parser.add_argument('--record_gif',default  = True ,type = bool)
    parser.add_argument('--pose_data_source',default  = "positions" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    parser.add_argument('--save_eval_data',default  = True ,type = bool,
                        help='save evaluation data to a pickle file')
    
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot_path
    record_gif_flag = args.record_gif
    log = []
    
    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )
    print("[INF] Plotting Flag:   " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    log.append("[INF] Dataset Name:    " + dataset)
    log.append("[INF] Sequence Name:   " + str(seq) )
    log.append("[INF] Plotting Flag:   " + str(plotting_flag))
    log.append("[INF] record gif Flag: " + str(record_gif_flag))
 
    ground_truth = {'pos_range': 2,
                    'warmupitrs': 300, # Number of frames to ignore at the beguinning
                    'roi': 200,
                    'anchor_range': 0}
    
    
    # log ground truth data 
    log.append("[INF] Ground Truth Parameters: ")
    log.append("[INF] Positive Range: " + str(ground_truth['pos_range']))
    log.append("[INF] Warmup Iterations: " + str(ground_truth['warmupitrs']))
    log.append("[INF] ROI: " + str(ground_truth['roi']))
    log.append("[INF] Anchor Range: " + str(ground_truth['anchor_range']))
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    print("[INF] Root directory: %s\n"% root_dir)
    log.append("[INF] Root directory: %s\n"% root_dir)
    
    dir_path = os.path.join(root_dir,dataset,seq)
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path
    
    print("[INF] Loading data from directory: %s\n" % dir_path)
    log.append("[INF] Loading data from directory: %s\n" % dir_path)
    
    save_root_dir  = os.path.join(root_dir,dataset,seq)
    if args.debug_mode:
        save_root_dir = os.path.join("temp",dataset,seq)
    
    os.makedirs(save_root_dir,exist_ok=True)

    print("[INF] Saving data to directory: %s\n" % save_root_dir)
    log.append("[INF] Saving data to directory: %s\n" % save_root_dir)
    
    # Create Save Directory
    save_root_dir  = os.path.join(root_dir,dataset,seq,"eval")
    if args.debug_mode:
        save_root_dir = os.path.join('temp',dataset,seq,"eval")
    
    os.makedirs(save_root_dir,exist_ok=True)

    # Loading DATA
    from dataloader.kitti.dataset import load_positions
    from dataloader.utils import gen_gt_constrained_by_rows,rotate_poses
    from dataloader import row_segments
    
    assert args.pose_data_source in ['gps','poses','positions'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    xy = load_positions(pose_file)

    print("[INF] Reading poses from: %s"% pose_file)
    print("[INF] Number of poses: %d"% xy.shape[0])
    log.append("[INF] Reading poses from: %s"% pose_file)
    log.append("[INF] Number of poses: %d"% xy.shape[0])
    
    # Rotate poses to align with the ROI frame
   
    rectangle_rois = load_row_bboxs(seq)

    anchors,positives,negatives  = gen_gt_constrained_by_rows(xy,rectangle_rois,**ground_truth)

    n_points = xy.shape[0]

    positive_range_str = str(ground_truth['pos_range'])

    print("="*30)
    print("[INF] Number of points: " + str(n_points))
    print("[INF] Number of anchors: " + str(len(anchors)))
    print("[INF] Number of positives: " + str(len(positives)))
    print("[INF] Number of negatives: " + str(len(negatives)))
    print("="*30)
    log.append("="*30)
    log.append("[INF] Number of points: " + str(n_points))
    log.append("[INF] Number of anchors: " + str(len(anchors)))
    log.append("[INF] Number of positives: " + str(len(positives)))
    log.append("[INF] Number of negatives: " + str(len(negatives)))
    log.append("="*30)
    
    # Generate Ground-truth Table
    if args.save_eval_data:
        
        ground_truth_name = f'ground_truth_loop_range_{positive_range_str}m'
        file = os.path.join(save_root_dir,ground_truth_name +'.pkl')

         # save the numpy arrays to a file using pickle
        with open(file, 'wb') as f:
            pickle.dump({
                'anchors': anchors,
                'positives': positives,
            }, f)
    
        print("[INF] saved ground truth at:" + file)
        log.append("[INF] saved ground truth at:" + file)
        


    if plotting_flag == True:
        flatten_pos = []
        for pos in positives:
             flatten_pos.extend(pos.flatten())
        
        flatten_pos = np.unique(np.array(flatten_pos))
        fig, ax = plt.subplots()
        ax.scatter(xy[:,0],xy[:,1],s=1,c='black',label='path')
        ax.scatter(xy[flatten_pos,0],xy[flatten_pos,1],s=1,c='green',label='positive')
        ax.scatter(xy[anchors,0],xy[anchors,1],s=1,c='blue',label='anchor')
        ax.set_aspect('equal')
        plt.xlabel('m')
        plt.ylabel('m')
        # set axis limits
        #plt.xlim(0, 100)
        
        file = os.path.join(save_root_dir,f'anchor_positive_pair_{positive_range_str}.png')
        # show the legend
        #ax.legend()
        plt.savefig(file)
        #ax.legend()
        plt.show()
        
        print("[INF] Plotting path and ground truth")
        print("[INF] Saving plot to: %s"% file)
        log.append("[INF] Plotting path and ground truth")
        log.append("[INF] Saving plot to: %s"% file)


    if record_gif_flag:
        n_points = xy.shape[0]
        # Generate Ground-truth Table 
        # Rows: Anchors
        table = np.zeros((n_points,n_points))
        for anchor,positive in zip(anchors,positives):
            table[anchor,positive] = 1

        print("[INF] Number of points: " + str(n_points))
        gif_file = os.path.join(save_root_dir,f'anchor_positive_pair_{positive_range_str}.gif')
        viz_overlap(xy,table,
                    record_gif = True,
                    file_name  = gif_file,
                    frame_jumps= 100)

        print("[INF] Saving gif to: %s"% gif_file)
        log.append("[INF] Saving gif to: %s"% gif_file)
        
    # Save log file
    log_file = os.path.join(save_root_dir,f'log_{positive_range_str}.txt')
    log_tex = '\n'.join(log)
    with open(log_file, 'w') as f:
        f.write(log_tex)
    print("[INF] Saving log to: %s"% log_file)  
    
        


    

