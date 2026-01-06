
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
    required_cols = ['x', 'y', 'z', 'timestamp', 'ID', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

def load_pose_easy(df, swap_xy=False, flip_z=False, include_label=True):
    """
    Convert DLO poses to GPS format.
    
    Args:
        dlo_df: DataFrame with DLO pose data
        ref_lat, ref_lon, ref_alt: Reference GPS coordinates (alt in mm)
        base_time: Base timestamp for the output
        start_seq: Starting sequence number
        rotation_angle: Angle to rotate poses before converting (radians)
        pose_frame: 'enu' or 'ned' - the coordinate frame of the poses
        swap_xy: If True, swap X and Y axes before processing
        flip_z: If True, flip the Z axis sign
        include_label: If True, include the label column in output
    
    Returns:
        DataFrame with GPS format data
    """
    # Detect input format
    input_format = detect_input_format(df)
    print(f"Detected input format: {input_format}")
    
    # If GPS times are provided, use them for nearest sync
    gps_times = None
    if hasattr(df, 'nearest_gps_times') and df.nearest_gps_times is not None:
        gps_times = df.nearest_gps_times

    for idx, row in df.iterrows():
        # Extract position based on format
        if input_format == 'path_easy':
            x_raw = row['x']
            y_raw = row['y']
            z_raw = row['z']
            timestamp = int(row['timestamp'] * 1e9)  # Default: nanoseconds
            if gps_times is not None:
                timestamp = gps_times[idx]
            pose_id = row['ID']
            label = row.get('label', 0)
        elif input_format == 'dlo_pose':
            x_raw = row['field.pose.pose.position.x']
            y_raw = row['field.pose.pose.position.y']
            z_raw = row['field.pose.pose.position.z']
            timestamp = row.get('%time', row.get('field.header.stamp', base_time))
            if gps_times is not None:
                timestamp = gps_times[idx]
            pose_id = idx
            label = 0
        else:  # dlo_velo or simple x,y,z
            x_raw = row['x']
            y_raw = row['y']
            z_raw = row['z']
            timestamp = row.get('%time', row.get('field.header.stamp', base_time))
            if gps_times is not None:
                timestamp = gps_times[idx]
            pose_id = idx
            label = row.get('label', 0)
        
        # Apply optional axis swapping/flipping
        if swap_xy:
            x_raw, y_raw = y_raw, x_raw
        if flip_z:
            z_raw = -z_raw
        
        # Apply rotation
        x_rotated = cos_a * x_raw - sin_a * y_raw
        y_rotated = sin_a * x_raw + cos_a * y_raw
        z_rotated = z_raw
        
        
        # Create GPS row with same structure as rawgps.csv
        covariance = 0.01
        
        
        if include_label:
            df['label'] = label
        
    return pd.DataFrame(gps_rows)


def parse_csv_file(df):
    """
    Parse the input CSV file into a structured format.
    """
    # Detect the input format
    input_format = detect_input_format(df)
    if input_format == 'path_easy':
        return load_path_easy(df)
    elif input_format == 'dlo_pose':
        return load_dlo_pose(df)
    elif input_format == 'dlo_velo':
        return load_dlo_velo(df)
    else:
        raise ValueError(f"Unknown input format: {input_format}")


class file_structure():
    
    def __init__(self,root,seq,lidar="pcd",verbose=False):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        self.target_dir = os.path.join(root, seq)
        assert os.path.isdir(self.target_dir),'target dataset does not exist: ' + self.target_dir

        # Get pose file
        # READ Poses from CSV file
        file_seq = seq.replace("PCD_", "")
        pose_csv_file = os.path.join(self.target_dir,"path_{}".format(file_seq).lower() + ".csv")

        assert os.path.isfile(pose_csv_file), 'pose file does not exist: ' + pose_csv_file
        print(f"Loading pose data from {pose_csv_file}...")
        self.df = load_from_csv(pose_csv_file)

        # Detect and print format
        input_format = detect_input_format(self.df)
        if input_format == 'path_easy':
            print("Detected path_easy.csv format")
            if 'label' in self.df.columns:
                print(f"Label distribution: {self.df['label'].value_counts().sort_index().to_dict()}")

        # Get point cloud files
        point_cloud_dir = os.path.join(self.target_dir,lidar)
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir

        
        pcl_files = sorted(Path(point_cloud_dir).glob('*.pcd'))
        
        ## Add column to df with a path to pcd for each pose
        self.df['pcd_path'] = self.df['ID'].apply(lambda x: pcl_files[x] if x < len(pcl_files) else None)


        if verbose:
            print("[INF] Found %d point cloud files in %s" %(len(self.point_cloud_files),point_cloud_dir))

    def _get_gps_timestamps_(self):
        """
        Get timestamps from the pose data.
        """
        return(self.df['timestamp'].values)
    
    def _get_pcl_timestamps_(self):
        raise NotImplementedError("PCL timestamps not implemented yet.")
    
    def _get_point_cloud_file_(self,idx=None):
        """
        Get point cloud file(s) for the given index.
        """
        return self.df['pcd_path'][idx].values

    def _get_pose_(self):
        "extract pose data from CSV"

        poses = self.df[['x','y','z','qx','qy','qz','qw']].values
        
        return(poses)

    def _get_target_dir(self):
        return(self.target_dir)
    
    def _get_row_labels(self):
        return(self.row_labels)



        
