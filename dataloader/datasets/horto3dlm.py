
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import os

import pickle
import numpy as np
from dataloader.utils import get_files

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)




def load_gps_to_RAM(file:str,local_frame=False)->np.ndarray:
    """ Loading the gps file to RAM.
    The gps file as the default structure of KITTI gps.txt file:
    Each line contains the following values separated by spaces:
    3D location (latitude, longitude, altitude)

    Args:
        file (str): default gps.txt file

    Returns:
        positions (np.array): array of positions (x,y,z) of the path (N,3)
        Note: N is the number of data points, No orientation is provided
    """
    assert os.path.isfile(file),"GPS file does not exist: " + file
    pose_array = []
    for line in open(file):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        if len(values) < 3:
            values = np.append(values,np.zeros(3-len(values)))
        elif len(values) > 3:
            values = values[:3]
        pose_array.append(values)

    pose_array = np.array(pose_array)
    return(pose_array)


def load_pose_to_RAM(file:str)->np.ndarray: 
    """ Loading the poses file to RAM. 
    The pose file as the default structure of KITTI poses.txt file:
    In each line, there are 16 values, which are the elements of a 4x4 Transformation matrix
    
    The elements are separated by spaces, and the values are in row-major order.

    T = [R11 R12 R13 tx   ->  [R11,R12,R13,tx,R21,R22,R23,ty,R31,R32,R33,tz,0,0,0,1]
         R21 R22 R23 ty
         R31 R32 R33 tz
         0   0   0   1 ]

    Args:
        file (str): default poses.txt file

    Returns:
        positions (np.array): array of positions (x,y,z) of the path (N,3)
        Note: N is the number of data points, No orientation is provided
    """
    assert os.path.isfile(file),"pose file does not exist: " + file
    pose_array = []
    for line in open(file):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        coordinates = np.array(values).reshape(4,4)
        pose_array.append(coordinates[:3,3])

    pose_array = np.array(pose_array)   
    return(pose_array)


def load_positions(file):
    if  "poses" in file:
        poses = load_pose_to_RAM(file)
    elif "gps" in file or "positions" in file:
        poses = load_gps_to_RAM(file)
    else:
        raise Exception("Invalid pose data source")
    return poses


class file_structure():
    
    def __init__(self,root,dataset,sequence,position_file="positions",verbose=False):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        #self.target_dir = os.path.join(root,dataset,sequence)
        self.target_dir = os.path.join(root,dataset,sequence,'extracted')
        assert os.path.isdir(self.target_dir),'target dataset does not exist: ' + self.target_dir

        # Get pose file
        pose_file = os.path.join(self.target_dir,position_file+'.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_positions(pose_file)
        
        # Get point cloud files
        point_cloud_dir = os.path.join(self.target_dir,'point_cloud')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        self.file_names, self.point_cloud_files = get_files(point_cloud_dir)

        # Get timestamps files
        self.pcl_timestamp = None
        pcl_timestamp_file = os.path.join(self.target_dir,'point_cloudtimestamp.txt')
        if os.path.isfile(pcl_timestamp_file):
            # open file and read the content in a list
            with open(pcl_timestamp_file, 'r') as f:
                self.pcl_timestamp = f.readlines()
                self.pcl_timestamp = [float(x.strip()) for x in self.pcl_timestamp]
        
        self.gps_timestamp = None
        gps_timestamp_file = os.path.join(self.target_dir,f'{position_file}_timestamp.txt') 
        if os.path.isfile(gps_timestamp_file):
            # open file and read the content in a list
            with open(gps_timestamp_file, 'r') as f:
                self.gps_timestamp = f.readlines()
                self.gps_timestamp = [float(x.strip()) for x in self.gps_timestamp]
        
        # Get row labels
        row_label_file = os.path.join(self.target_dir,'point_row_labels.pkl')
        assert os.path.isfile(row_label_file), "Row label file does not exist " + row_label_file
        with open(row_label_file, 'rb') as f:
            self.row_labels = pickle.load(f)

        if verbose:
            print("[INF] Loading poses from: %s"% pose_file)
            print("[INF] Found %d poses in %s" %(len(self.pose),pose_file))
            print("[INF] Found %d point cloud files in %s" %(len(self.point_cloud_files),point_cloud_dir))

    def _get_gps_timestamps_(self):
        return(self.gps_timestamp)
    
    def _get_pcl_timestamps_(self):
        return(self.pcl_timestamp)
    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files,self.file_names)
        return(self.point_cloud_files[idx],self.file_names[idx])
    
    def _get_pose_(self):
        return(self.pose)

    def _get_target_dir(self):
        return(self.target_dir)
    
    def _get_row_labels(self):
        return(self.row_labels)



        
