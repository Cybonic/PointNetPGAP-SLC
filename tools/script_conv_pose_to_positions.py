import os, sys
import numpy as np
import argparse
import utm  # pip install utm
import yaml
import matplotlib.pyplot as plt

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

from dataloader.horto3dlm.utils import save_positions_KITTI_format
from dataloader.utils import rotate_poses
from dataloader.horto3dlm.dataset import load_positions

def conv_to_positions(poses,map_local_frame = False,rotation_angle= 0):
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
    pose_array = poses[:,:3]
    if map_local_frame:
        # Convert GPS coordinates to UTM
        pose_array = pose_array.copy()
        utm_gps = utm.from_latlon(pose_array[:,1], pose_array[:,0])
        pose_array[:,0] = utm_gps[0]
        pose_array[:,1] = utm_gps[1]
        
    pose_array = rotate_poses(pose_array.copy(),rotation_angle)
    pose_array =pose_array - pose_array[0,:]
        
    return pose_array


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/SHARE/DATASET')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'GreenHouse',
                                    type= str,
                                    help='dataset root directory.'
                                    )
    
    parser.add_argument('--seq',default  = "e3/extracted",type = str)
    parser.add_argument('--pose_data_source',default  = "poses" ,type = str, choices = ['gps','poses'])
    parser.add_argument('--debug_mode',default  = False ,type = bool, 
                        help='debug mode, when turned on the files saved in a temp directory')
    parser.add_argument('--save_data',default  = True ,type = bool,
                        help='save evaluation data to a pickle file')
    parser.add_argument('--rot_anlge',default  = 88 ,type = int,
                        help='rotation angle in degrees to rotate the path. the path is rotated at the goemtrical center, ' + 
                        "positive values rotate anti-clockwise, negative values rotate clockwise")
    
    log = []
    args = parser.parse_args()
    root    = args.root
    dataset = args.dataset 
    seq     = args.seq


    print("[INF] Dataset Name:    " + dataset)
    print("[INF] Sequence Name:   " + str(seq) )

    log.append("[INF] Dataset Name:    " + dataset)
    log.append("[INF] Sequence Name:   " + str(seq) )
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions',f'{dataset}.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    log.append("Opening session config file: %s" % session_cfg_file)
    
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    print("[INF] Root directory: %s\n"% root_dir)
    log.append("[INF] Root directory: %s\n"% root_dir)
    
    
    dir_path = os.path.join(root_dir,dataset,seq)
    assert os.path.exists(dir_path), "Data directory does not exist:" + dir_path
    
    print("[INF] Loading data from directory: %s\n" % dir_path)
    log.append("[INF] Loading data from directory: %s\n" % dir_path)
    
    save_root_dir  = dir_path
    if args.debug_mode:
        save_root_dir = os.path.join("temp",dataset,seq)
        os.makedirs(save_root_dir,exist_ok=True)

    print("[INF] Saving data to directory: %s\n" % save_root_dir)
    log.append("[INF] Saving data to directory: %s\n" % save_root_dir)
    
    assert args.pose_data_source in ['gps','poses'], "Invalid pose data source"
    pose_file = os.path.join(dir_path,f'{args.pose_data_source}.txt')
    
    poses     = load_positions(pose_file)
    
    map_local_frame = True
    if not 'gps' in args.pose_data_source:
        map_local_frame = False
        
    positions = conv_to_positions(poses,map_local_frame = map_local_frame, rotation_angle= args.rot_anlge)

    # plot the positions in the local frame
    plt.figure()
    plt.plot(positions[:,0],positions[:,1])
    plt.grid(True)
    #plt.axis('equal')
    
    x_min = positions[:,0].min() - 5
    x_max = positions[:,0].max() + 5
    y_min = positions[:,1].min() - 5 
    y_max = positions[:,1].max() + 5
    
    # limit the axis 
    plt.xlim([x_min,x_max])
    plt.ylim([y_min,y_max])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Positions in the local frame')
    plt.savefig(os.path.join(save_root_dir,'positions.png'))
    
    log.append("Saving positions to: %s"% os.path.join(save_root_dir,'positions.png'))
    log.append("Rotation angle: %d"%(args.rot_anlge))
    
    if  args.save_data:
        file = os.path.join(save_root_dir,'positions.txt')
        save_positions_KITTI_format(save_root_dir,positions)
        log.append("Saving positions to: %s"% file)
    
    
    print("****************************************************\n")
    
    terminal_log = '\n'.join(log)
    # Save log to a file
    log_file = os.path.join(save_root_dir,'log.txt')
    with open(log_file, 'w') as f:
        f.write(terminal_log)
        
    #print(terminal_log)
    
 


    

