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


    

def generate_label_colors(num_colors):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20', num_colors)
    indices = np.arange(num_colors)
    np.random.seed(42)  # For reproducibility, remove or change for different results
    np.random.shuffle(indices)
    colors = {}
    for i, idx in enumerate(indices):
        colors[i] = cmap(idx)
    return colors


def hex_color_to_rgb(color_str):
    # KML: AABBGGRR
    bb = int(color_str[2:4], 16)
    gg = int(color_str[4:6], 16)
    rr = int(color_str[6:8], 16)
    return (rr/255, gg/255, bb/255)

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

def label_color_rgb(label):
    """Get RGB color for a given label."""
    hex_color = get_label_color(label)
    return hex_color_to_rgb(hex_color)

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


def load_path_easy(df: pd.DataFrame):
    """
    Load path_easy.csv format with columns:
    secs, nsecs, timestamp, ID, frame_id, x, y, z, qx, qy, qz, qw, label

    Input:
    df: A pandas DataFrame containing the path_easy.csv data.

    Output:
    A pandas DataFrame with the loaded path_easy data.
    """
    # Clean column names
    df.columns = df.columns.str.strip().str.replace('"', '')
    
    print(f"Loaded path_easy format with columns: {list(df.columns)}")
    
    # Validate required columns
    required_cols = ['x', 'y', 'z', 'timestamp', 'ID', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    if 'label' in df.columns:
        print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    return df



def parse_csv_file(filepath: str) -> pd.DataFrame:
    """
    Parse the input CSV file into a structured format.
    """
    df = pd.read_csv(filepath, comment='/')
    # Detect the input format
    input_format = detect_input_format(df)
    if input_format == 'path_easy':
        return load_path_easy(df)
    else:
        raise ValueError(f"Unknown input format: {input_format}")

def compute_loop(df: pd.DataFrame):
    """
    Compute the loop closure for the given dataframe.
    """
    # Placeholder for loop closure computation
    print(f"Computing loop closure for dataframe with {len(df)} rows.")
    
    
    return df

class file_structure():
    
    def __init__(self,root,seq,lidar="pcd",verbose=False):
        # assert isinstance(sequences,list)
        self.pose = []
        self.point_cloud_files = []
        self.target_dir = []

        self.target_dir = os.path.join(root, seq)
        if verbose: print(f"Checking target directory at: {self.target_dir}")
        assert os.path.isdir(self.target_dir),'target dataset does not exist: ' + self.target_dir

        # Get pose file
        # READ Poses from CSV file
        file_seq = seq.replace("PCD_", "")
        pose_csv_file = os.path.join(self.target_dir,"path_{}".format(file_seq).lower() + ".csv")
        if verbose: print(f"Checking pose file at: {pose_csv_file}")
        assert os.path.isfile(pose_csv_file), 'pose file does not exist: ' + pose_csv_file
        if verbose: print(f"Loading pose data from {pose_csv_file}...")
        self.df = parse_csv_file(pose_csv_file)

        # Get point cloud files
        point_cloud_dir = os.path.join(self.target_dir,lidar)
        assert os.path.isdir(point_cloud_dir),'point cloud dir does not exist: ' + point_cloud_dir
        def extract_number(p):
            return int(p.stem) if p.stem.isdigit() else p.stem
        
        pcl_files = sorted(Path(point_cloud_dir).glob('*.pcd'), key=extract_number)
        
        ## Add column to df with a path to pcd for each pose
        self.df['pcd_path'] = self.df['ID'].apply(lambda x: pcl_files[x] if x < len(pcl_files) else None)

        if verbose: print("[INF] Found %d point cloud files in %s" %(len(self.point_cloud_files),point_cloud_dir))

    def _get_timestamps_(self):
        """
        Get timestamps from the pose data.
        """
        return(self.df['timestamp'].values)

    def _get_timestamp_(self,i):
        """
        Get timestamps from the pose data.
        """
        return(self.df['timestamp'].values[i])

    def _get_pcl_timestamps_(self):
        raise NotImplementedError("PCL timestamps not implemented yet.")

    def _get_point_cloud_files_(self)->np.ndarray:
        """
        Get point cloud file(s) for the given index.
        """
        return self.df['pcd_path'].values

    def _get_point_cloud_file_(self,i)->str:
        """
        Get point cloud file(s) for the given index.
        """
        return self.df['pcd_path'].values[i]

    def _get_pose_(self)->np.ndarray:
        "extract pose data from CSV"
        return self.df[['x','y','z','qx','qy','qz','qw']].values

    def _get_pose_(self,i:int)->np.ndarray:
        """
        Get the pose for a specific index.
        """
        return self.df[['x','y','z','qx','qy','qz','qw']].values[i]

    def _get_positions_(self,)->np.ndarray:
        """
        Get the position for a specific index.
        """
        return self.df[['x','y','z']].values

    def _get_position_(self,i:int)->np.ndarray:
        """
        Get the position for a specific index.
        """
        return self.df[['x','y','z']].values[i]

    def _get_orientation_(self,i:int)->np.ndarray:
        """
        Get the orientation for a specific index.
        """
        return self.df[['qx','qy','qz','qw']].values[i]
    

    def _get_target_dir(self)->str:
        """
        Get the target directory.
        """
        return self.target_dir

    def _get_labels(self)->np.ndarray:
        """
        Get all labels.
        """
        return(self.df['label'].values)

    def _get_label_(self,i:int)->str:
        """
        Get the label for a specific index.
        """
        return(self.df['label'].values[i])

    def _get_frame_ids(self) -> np.ndarray:
        """
        Get all frame IDs.
        """
        return self.df['ID'].values

    def _get_frame_id_(self, i: int) -> str:
        """
        Get the frame ID for a specific index.
        """
        return self.df['ID'].values[i]

    def _load_pcd_(self, i:int)->np.ndarray:
        """
        Load the point cloud data for a specific index.
        """
        pcd_path = self._get_point_cloud_file_(i)
        assert os.path.isfile(pcd_path), 'point cloud file does not exist: ' + pcd_path
        return self._load_pcd_file(pcd_path)

   
    def compute_nearest_neighbor_label(self, position_idx: int) -> dict:
        """
        Find the nearest neighbor of a given position with a different label.
        
        Args:
            position_idx: Index of the query position
            
        Returns:
            Dictionary with keys:
                - 'neighbor_idx': Index of nearest neighbor
                - 'neighbor_label': Label of nearest neighbor
                - 'query_label': Label of query position
                - 'distance': Euclidean distance to nearest neighbor
                - 'position': Position of query point
                - 'neighbor_position': Position of nearest neighbor
        """
        if position_idx < 0 or position_idx >= len(self.df):
            raise ValueError(f"Invalid position index: {position_idx}")
        
        # Get query position and label
        query_pos = self._get_position_(position_idx)
        query_label = self._get_label_(position_idx)
        
        # Get all positions and labels
        all_positions = self._get_positions_()
        all_labels = self._get_labels()
        
        # Find positions with different labels
        different_label_mask = all_labels == query_label
        different_label_indices = np.where(different_label_mask)[0]
        
        if len(different_label_indices) == 0:
            return {
                'neighbor_idx': None,
                'neighbor_label': None,
                'query_label': query_label,
                'distance': np.inf,
                'position': query_pos,
                'neighbor_position': None
            }
        
        # Compute distances to all positions with different labels
        different_positions = all_positions[different_label_indices]
        distances = np.linalg.norm(different_positions - query_pos, axis=1)
        
        # Find nearest neighbor
        nearest_idx_in_subset = np.argmin(distances)
        nearest_idx = different_label_indices[nearest_idx_in_subset]
        nearest_distance = distances[nearest_idx_in_subset]
        
        return {
            'neighbor_idx': nearest_idx,
            'neighbor_label': all_labels[nearest_idx],
            'query_label': query_label,
            'distance': nearest_distance,
            'position': query_pos,
            'neighbor_position': all_positions[nearest_idx]
        }

    def compute_nearest_neighbor_different_frame(self, position_idx: int) -> dict:
        """
        Find the nearest neighbor of a given position with a different frame ID but same label.
        
        Args:
            position_idx: Index of the query position
            
        Returns:
            Dictionary with keys:
                - 'neighbor_idx': Index of nearest neighbor
                - 'neighbor_frame': Frame ID of nearest neighbor
                - 'query_frame': Frame ID of query position
                - 'query_label': Label of query position
                - 'distance': Euclidean distance to nearest neighbor
                - 'position': Position of query point
                - 'neighbor_position': Position of nearest neighbor
        """
        if position_idx < 0 or position_idx >= len(self.df):
            raise ValueError(f"Invalid position index: {position_idx}")
        
        # Get query position, frame ID, and label
        query_pos = self._get_position_(position_idx)
        query_frame = self._get_frame_id_(position_idx)
        query_label = self._get_label_(position_idx)
        
        # Get all positions, frame IDs, and labels
        all_positions = self._get_positions_()
        all_frames = self._get_frame_ids()
        all_labels = self._get_labels()
        
        # Find positions with SAME label but DIFFERENT frame ID
        same_label_mask = all_labels == query_label
        different_frame_mask = all_frames != query_frame
        combined_mask = same_label_mask & different_frame_mask
        different_frame_indices = np.where(combined_mask)[0]
        
        if len(different_frame_indices) == 0:
            return {
                'neighbor_idx': None,
                'neighbor_frame': None,
                'query_frame': query_frame,
                'query_label': query_label,
                'distance': np.inf,
                'position': query_pos,
                'neighbor_position': None
            }
        
        # Compute distances to all positions with same label and different frame ID
        different_positions = all_positions[different_frame_indices]
        distances = np.linalg.norm(different_positions - query_pos, axis=1)
        
        # Find nearest neighbor
        nearest_idx_in_subset = np.argmin(distances)
        nearest_idx = different_frame_indices[nearest_idx_in_subset]
        nearest_distance = distances[nearest_idx_in_subset]
        
        return {
            'neighbor_idx': nearest_idx,
            'neighbor_frame': all_frames[nearest_idx],
            'query_frame': query_frame,
            'query_label': query_label,
            'distance': nearest_distance,
            'position': query_pos,
            'neighbor_position': all_positions[nearest_idx]
        }

    def compute_all_nearest_neighbors_different_frame(self) -> list:
        """
        Compute nearest neighbor with same label but different frame ID for all positions.
        
        Returns:
            List of dictionaries, one for each position
        """
        results = []
        for i in range(len(self.df)):
            result = self.compute_nearest_neighbor_different_frame(i)
            results.append(result)
        return results
