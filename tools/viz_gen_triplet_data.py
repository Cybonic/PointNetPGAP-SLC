
import os, sys
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import pickle
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from utils.viz import myplot

from dataloader.utils import rotate_poses,gen_gt_constrained_by_rows
from dataloader.kitti.kitti_dataset import load_positions
from dataloader import row_segments

def viz_triplet(xy, loops, record_gif= False, file_name = 'anchor_positive_pair.gif',frame_jumps=50):
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
    mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'k')
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
        color[negatives_idx] = 'r'
        scale[negatives_idx] = 100

        x = xy[:i+1,1]
        y = xy[:i+1,0]

        mplot.update_plot(y,x,offset=2,color=color,zoom=0,scale=scale) 



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/deep/Dropbox/SHARE/DATASET', help='path to the data directory')
    parser.add_argument('--dataset',
                                    default = 'GreenHouse',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    parser.add_argument('--seq',default  = "e3/extracted",type = str, 
                        help='path to the data of the sequence')
    parser.add_argument('--pose_data_source',default  = "positions" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--show_static_plot',default  = True ,type = bool)
    parser.add_argument('--record_gif',default  = True ,type = bool)
    parser.add_argument('--save_triplet_data',default  = True ,type = bool)
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files are saved at temp/.../... ')
    
    args = parser.parse_args()

    root_dir    = args.root
    dataset = args.dataset 
    sequence     = args.seq
    plotting_flag = args.show_static_plot
    record_gif_flag = args.record_gif
    save_triplet_data = args.save_triplet_data
    log = []
    ground_truth = {'pos_range':2, # Loop Threshold [m]
                    'neg_range':10,
                    'num_neg':20,
                    'num_pos':1,
                    'warmupitrs': 200, # Number of frames to ignore at the beguinning
                    'roi':100, # Region of interest [m]
                    'anchor_range':0.5 # Range to search for the anchor
                    }
    
    # log ground truth parameters
    log.append("[INF] Ground Truth Parameters: ")
    log.append("[INF] pos_range: " + str(ground_truth['pos_range']))
    log.append("[INF] neg_range: " + str(ground_truth['neg_range']))
    log.append("[INF] num_neg:   " + str(ground_truth['num_neg']))
    log.append("[INF] num_pos:   " + str(ground_truth['num_pos']))
    log.append("[INF] warmup:    " + str(ground_truth['warmupitrs']))
    log.append("[INF] roi:       " + str(ground_truth['roi']))
    log.append("[INF] anchor_range: " + str(ground_truth['anchor_range']))
    
    print("****************************************************\n")
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("[INF] Opening session config file: %s\n" % session_cfg_file)
    log.append("[INF] Opening session config file: %s\n" % session_cfg_file)
    
    if root_dir == 'None':
        device_name = os.uname()[1] # Get the device name
        print("Device name: %s" % device_name)
        log.append("Device name: %s" % device_name)
        pc_config_file = "sessions/pc_config.yaml" # Load the device config file
        assert os.path.exists(pc_config_file), "Device config file does not exist"
        
        pc_config = yaml.safe_load(open(pc_config_file, 'r')) # Load the device config file
        # Get the root directory based on the device name 
        root_dir = pc_config[device_name]

    print("****************************************************\n")
    print("[INF] Session Config: ")
    print("[INF] Root directory: %s "% root_dir) # pint all the session config equally spaced
    print("[INF] Dataset:      " + dataset)
    print("[INF] Sequence:     " + str(sequence) )
    print("[INF] record gif:   " + str(record_gif_flag))
    print("[INF] plot static:  " + str(plotting_flag))
    print("[INF] save triplet: " + str(save_triplet_data))
    print("[INF] debug mode:   " + str(args.debug_mode))
    print("[INF] pose source:  " + str(args.pose_data_source))
    
    log.append("[INF] Session Config: ")
    log.append("[INF] Root directory: %s "% root_dir) # pint all the session config equally spaced
    log.append("[INF] Dataset:      " + dataset)
    log.append("[INF] Sequence:     " + str(sequence) )
    log.append("[INF] record gif:   " + str(record_gif_flag))
    log.append("[INF] plot static:  " + str(plotting_flag))
    log.append("[INF] save triplet: " + str(save_triplet_data))
    log.append("[INF] debug mode:   " + str(args.debug_mode))
    log.append("[INF] pose source:  " + str(args.pose_data_source))
    
    
    print("****************************************************\n")
    print("[INF] Ground Truth Parameters: ")
    print("[INF] pos_range: " + str(ground_truth['pos_range']))
    print("[INF] neg_range: " + str(ground_truth['neg_range']))
    print("[INF] num_neg:   " + str(ground_truth['num_neg']))
    print("[INF] num_pos:   " + str(ground_truth['num_pos']))
    print("[INF] warmup:    " + str(ground_truth['warmupitrs']))
    print("[INF] roi:       " + str(ground_truth['roi']))
    print("****************************************************\n")
    print("[INF] Root directory: %s\n"% root_dir)

    dir_path = os.path.join(root_dir,dataset,sequence)
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path

    print("[INF] Loading data from directory: %s\n" % dir_path)
    log.append("[INF] Loading data from directory: %s\n" % dir_path)
    
    assert args.pose_data_source in ['gps','poses','positions'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    xy = load_positions(pose_file)

    print("[INF] Reading poses from: %s"% pose_file)
    log.append("[INF] Reading poses from: %s"% pose_file)
    print("[INF] Number of poses: %d"% xy.shape[0])
    log.append("[INF] Number of poses: %d"% xy.shape[0])

    from viz_row_labels import load_row_bboxs
    
    rectangle_rois = load_row_bboxs(sequence)
    
    anchors,positives,negatives  = gen_gt_constrained_by_rows(xy,rectangle_rois,**ground_truth)

    n_points = xy.shape[0]
    anchor_range_str   = str(ground_truth['anchor_range'])
    negative_range_str = str(ground_truth['neg_range'])
    positive_range_str = str(ground_truth['pos_range'])
    # save the ground truth
    ground_truth_name = f'ground_truth_ar{anchor_range_str}m_nr{negative_range_str}m_pr{positive_range_str}m'

    save_root_dir  = os.path.join(root_dir,dataset,sequence,"triplet")
    if args.debug_mode:
        save_root_dir = os.path.join('temp',dataset,sequence,"triplet")
    
    os.makedirs(save_root_dir,exist_ok=True)

    print("[INF] Saving data to directory: %s\n" % save_root_dir)
    log.append("[INF] Saving data to directory: %s\n" % save_root_dir)
    
    file = os.path.join(save_root_dir,ground_truth_name +'.pkl')
    
    if save_triplet_data:
         # save the numpy arrays to a file using pickle
        with open(file, 'wb') as f:
            pickle.dump({
                'anchors': anchors,
                'positives': positives,
                'negatives': negatives
            }, f)
    
        print("[INF] saved ground truth at:" + file)
        log.append("[INF] saved ground truth at:" + file)


    # Generate Ground-truth Table 
    # Rows: Anchors
    table = np.zeros((n_points,n_points))
    for anchor,positive,negative in zip(anchors,positives,negatives):
        table[anchor,positive] = 1
        table[anchor,negative] = -1
    
    print("[INF] Number of anchors: " + str(len(anchors)))
    log.append("[INF] Number of anchors: " + str(len(anchors)))
    # Plot the ground truth
    gif_file = os.path.join(save_root_dir,ground_truth_name + '.gif')

    if record_gif_flag:
        viz_triplet(xy,table,
                    record_gif=True,
                    file_name=gif_file,
                    frame_jumps=1)
        
        print("[INF] saved gif at:" + gif_file)
        log.append("[INF] saved gif at:" + gif_file)
    
    
    # save the log file
    log_file = os.path.join(save_root_dir,ground_truth_name + '.log')

    log_text = '\n'.join(log)
    with open(log_file, 'w') as f:
        f.write(log_text)
    


