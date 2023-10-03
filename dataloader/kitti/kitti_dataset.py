
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import os


import numpy as np
from dataloader.utils import get_files

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def load_pose_to_RAM(file):
    assert os.path.isfile(file),"pose file does not exist: " + file
    pose_array = []
    for line in open(file):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        coordinates = np.array(values).reshape(4,4)
        #position = values[[3,7,11]]
        #position[:,1:] =position[:,[2,1]] 
        pose_array.append(coordinates[:3,3])

    pose_array = np.array(pose_array)   
    #pose_array[:,1:] =pose_array[:,[2,1]] 
    return(pose_array)

class kittidataset():
    
    def __init__(self,root,dataset,sequence):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        #for seq in sequences:
        self.target_dir = os.path.join(root,dataset,sequence)
        #self.target_dir.append(target_dir)
        assert os.path.isdir(self.target_dir),'target dataset does nor exist: ' + self.target_dir

        pose_file = os.path.join(self.target_dir,'poses.txt')
        assert os.path.isfile(pose_file),'pose file does not exist: ' + pose_file
        self.pose = load_pose_to_RAM(pose_file)
        #self.pose.extend(pose)

        point_cloud_dir = os.path.join(self.target_dir,'point_cloud')
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        self.file_names, self.point_cloud_files = get_files(point_cloud_dir)

    
    def _get_point_cloud_file_(self,idx=None):
        if idx == None:
            return(self.point_cloud_files,self.file_names)
        return(self.point_cloud_files[idx],self.file_names[idx])
    
    def _get_pose_(self):
        return(self.pose)

    def _get_target_dir(self):
        return(self.target_dir)



        
