
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import pandas as pd
import numpy as np
from pathlib import Path
import numpy as np

# List of all timestamp names we might encounter
TIMESTAMP_COLUMNS = ['field.header.stamp', 'timestamp', '%timestamp']


# Default color palette for labels (AABBGGRR format for KML)
LABEL_COLORS = {
    0: 'ff00ff00',  # Green
    1: 'ff0000ff',  # Red
    2: 'ffff0000',  # Blue
    3: 'ff00ffff',  # Yellow
    4: 'ffff00ff',  # Magenta
    5: 'ffffff00',  # Cyan
    6: 'ff0080ff',  # Orange
    7: 'ff800080',  # Purple
    8: 'ff008080',  # Olive
    9: 'ff808000',  # Teal
}

LABEL_NAMES = {
    0: 'Label 0 (Green)',
    1: 'Label 1 (Red)',
    2: 'Label 2 (Blue)',
    3: 'Label 3 (Yellow)',
    4: 'Label 4 (Magenta)',
    5: 'Label 5 (Cyan)',
    6: 'Label 6 (Orange)',
    7: 'Label 7 (Purple)',
    8: 'Label 8 (Olive)',
    9: 'Label 9 (Teal)',
}


def load_from_csv(filepath):
    """Load DLO velocity/pose data from CSV."""
    df = pd.read_csv(filepath)
    return df


def get_label_color(label):
    """Get color for a given label."""
    if label is None:
        return 'ff0000ff'  # Default red
    try:
        label_int = int(label)
        return LABEL_COLORS.get(label_int, LABEL_COLORS[label_int % len(LABEL_COLORS)])
    except (ValueError, TypeError):
        return 'ff0000ff'  # Default red


def detect_input_format(df):
    """
    Detect the format of the input dataframe.
    Returns: 'path_easy', 'dlo_velo', or 'dlo_pose'
    """
    columns = set(df.columns)
    
    # path_easy format: secs, nsecs, timestamp, ID, frame_id, x, y, z, qx, qy, qz, qw, label
    if {'secs', 'nsecs', 'timestamp', 'ID', 'x', 'y', 'z', 'label'}.issubset(columns):
        return 'path_easy'
    
    # dlo_velo format with field.pose.pose.position
    if 'field.pose.pose.position.x' in columns:
        return 'dlo_pose'
    
    # Simple x, y, z format
    if {'x', 'y', 'z'}.issubset(columns):
        return 'dlo_velo'
    
    return 'unknown'


def load_path_easy(filepath):
    """
    Load path_easy.csv format with columns:
    secs, nsecs, timestamp, ID, frame_id, x, y, z, qx, qy, qz, qw, label
    """
    df = pd.read_csv(filepath, comment='/')
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    print(f"Loaded path_easy format with columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['x', 'y', 'z', 'timestamp', 'ID']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df




class file_structure():
    
    def __init__(self,target_dir,lidar="pcd",verbose=False):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        #self.target_dir = os.path.join(root,dataset,sequence)
        self.target_dir =target_dir
        assert os.path.isdir(self.target_dir),'target dataset does not exist: ' + self.target_dir

        # Get pose file
        # READ Poses from CSV file
        pose_file = os.path.join(self.target_dir,pose_file)
        assert os.path.isfile(pose_file), 'pose file does not exist: ' + pose_file
        print(f"Loading pose data from {pose_file}...")
        pose_df = load_from_csv(pose_file)
        
        # Detect and print format
        input_format = detect_input_format(pose_df)
        if input_format == 'path_easy':
            print("Detected path_easy.csv format")
            if 'label' in pose_df.columns:
                print(f"Label distribution: {pose_df['label'].value_counts().sort_index().to_dict()}")

        # Get point cloud files
        point_cloud_dir = os.path.join(self.target_dir,lidar)
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir

        
        pcl_files = sorted(Path(point_cloud_dir).glob('*.pcd'))
        
        ## Add column to df with a path to pcd for each pose
        pose_df['pcd_path'] = pose_df['ID'].apply(lambda x: pcl_files[x] if x < len(pcl_files) else None)


        if verbose:
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



        
