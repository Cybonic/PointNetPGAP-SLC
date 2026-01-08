import os
import sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))

import json
import pandas as pd
import numpy as np
from pathlib import Path

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


def generate_label_colors(num_colors):
    """Generate a colormap for labels using matplotlib."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20', num_colors)
    indices = np.arange(num_colors)
    np.random.seed(42)
    np.random.shuffle(indices)
    colors = {}
    for i, idx in enumerate(indices):
        colors[i] = cmap(idx)
    return colors


def hex_color_to_rgb(color_str):
    """Convert KML color (AABBGGRR format) to RGB tuple."""
    bb = int(color_str[2:4], 16)
    gg = int(color_str[4:6], 16)
    rr = int(color_str[6:8], 16)
    return (rr/255, gg/255, bb/255)


def load_from_csv(filepath):
    """Load pose data from CSV file."""
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


def load_pcd_file(pcd_path: str, use_plyfile: bool = False) -> np.ndarray:
    """
    Load a PCD (Point Cloud Data) file.
    
    Supports both Open3D and plyfile methods.
    
    Args:
        pcd_path: Path to the PCD file
        use_plyfile: If True, use plyfile; otherwise use Open3D
        
    Returns:
        Numpy array of shape (N, 3) or (N, 4) containing point cloud data
        
    Examples:
        >>> points = load_pcd_file('data.pcd')
        >>> print(points.shape)  # (N, 3)
        
        >>> points = load_pcd_file('data.pcd', use_plyfile=True)
    """
    if not os.path.isfile(pcd_path):
        raise FileNotFoundError(f"PCD file not found: {pcd_path}")
    
    if use_plyfile:
        return _load_pcd_plyfile(pcd_path)
    else:
        return _load_pcd_open3d(pcd_path)


def _load_pcd_open3d(pcd_path: str) -> np.ndarray:
    """Load PCD file using Open3D library."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("open3d is required. Install with: pip install open3d")
    
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        points = np.asarray(pcd.points)
        
        # Include colors if available
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            points = np.hstack([points, colors])
        
        return points
    except Exception as e:
        raise ValueError(f"Failed to load PCD file with Open3D: {e}")


def _load_pcd_plyfile(pcd_path: str) -> np.ndarray:
    """Load PCD file using plyfile library (more memory efficient)."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("plyfile is required. Install with: pip install plyfile")
    
    try:
        ply_data = PlyData.read(pcd_path)
        vertex = ply_data['vertex']
        
        # Extract x, y, z coordinates
        points = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        
        return points
    except Exception as e:
        raise ValueError(f"Failed to load PCD file with plyfile: {e}")

class file_structure():
    
    def __init__(self, root, seq, lidar="pcd", verbose=False):
        """Initialize file_structure with dataset root and sequence name."""
        self.target_dir = os.path.join(root, seq)
        if verbose:
            print(f"Checking target directory at: {self.target_dir}")
        assert os.path.isdir(self.target_dir), 'target dataset does not exist: ' + self.target_dir

        # Load pose data from CSV file
        file_seq = seq.replace("PCD_", "")
        pose_csv_file = os.path.join(self.target_dir, f"path_{file_seq.lower()}.csv")
        if verbose:
            print(f"Checking pose file at: {pose_csv_file}")
        assert os.path.isfile(pose_csv_file), 'pose file does not exist: ' + pose_csv_file
        if verbose:
            print(f"Loading pose data from {pose_csv_file}...")
        self.df = parse_csv_file(pose_csv_file)

        # Load point cloud file paths
        point_cloud_dir = os.path.join(self.target_dir, lidar)
        assert os.path.isdir(point_cloud_dir), 'point cloud dir does not exist: ' + point_cloud_dir
        
        def extract_number(p):
            return int(p.stem) if p.stem.isdigit() else p.stem
        
        pcl_files = sorted(Path(point_cloud_dir).glob('*.pcd'), key=extract_number)
        
        # Add column to df with path to pcd for each pose
        self.df['pcd_path'] = self.df['ID'].apply(lambda x: pcl_files[x] if x < len(pcl_files) else None)

        if verbose:
            print(f"[INF] Found {len(pcl_files)} point cloud files in {point_cloud_dir}")

    def _get_timestamps_(self):
        """
        Get timestamps from the pose data.
        """
        return(self.df['timestamp'].values)

    def _get_timestamp_(self, i):
        """Get timestamp for a specific index."""
        return self.df['timestamp'].values[i]

    def _get_point_cloud_files_(self) -> np.ndarray:
        """Get all point cloud file paths."""
        return self.df['pcd_path'].values

    def _get_point_cloud_file_(self, i) -> str:
        """Get point cloud file path for a specific index."""
        return self.df['pcd_path'].values[i]

    def _get_pose_(self, i: int) -> np.ndarray:
        """Get the pose (position + orientation) for a specific index."""
        return self.df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].values[i]

    def _get_positions_(self) -> np.ndarray:
        """Get all positions (x, y, z coordinates)."""
        return self.df[['x', 'y', 'z']].values

    def _get_position_(self, i: int) -> np.ndarray:
        """Get the position (x, y, z) for a specific index."""
        return self.df[['x', 'y', 'z']].values[i]

    def _get_labels(self) -> np.ndarray:
        """Get all labels."""
        return self.df['label'].values

    def _get_label_(self, i: int) -> int:
        """Get the label for a specific index."""
        return self.df['label'].values[i]

    def _get_frame_ids(self) -> np.ndarray:
        """Get all frame IDs."""
        return self.df['ID'].values

    def _get_frame_id_(self, i: int) -> int:
        """Get the frame ID for a specific index."""
        return self.df['ID'].values[i]

    def _get_target_dir(self) -> str:
        """Get the target directory."""
        return self.target_dir

    def _load_pcd_(self, i: int) -> np.ndarray:
        """Load the point cloud data for a specific index."""
        pcd_path = self._get_point_cloud_file_(i)
        assert os.path.isfile(pcd_path), f'point cloud file does not exist: {pcd_path}'
        return load_pcd_file(str(pcd_path))

    def compute_nearest_neighbor_different_frame(self, position_idx: int, lower_bound_idx=50) -> dict:
        """
        Find the nearest neighbor of a given position with a different (past) frame ID but same label.
        Only searches in frames with lower IDs (past frames).
        
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
        if position_idx < lower_bound_idx or position_idx >= len(self.df):
            raise ValueError(f"Invalid position index: {position_idx}")
        
        # Get query position, frame ID, and label
        query_pos = self._get_position_(position_idx)
        query_frame = self._get_frame_id_(position_idx)
        query_label = self._get_label_(position_idx)
        
        # Get all positions, frame IDs, and labels
        all_positions = self._get_positions_()
        all_frames = self._get_frame_ids()
        all_labels = self._get_labels()
        
        # Find positions with SAME label, DIFFERENT frame ID, and PAST frames (lower ID)
        same_label_mask = all_labels == query_label
        past_frame_mask = all_frames < query_frame-lower_bound_idx
        combined_mask = same_label_mask & past_frame_mask
        past_frame_indices = np.where(combined_mask)[0]
        
        if len(past_frame_indices) == 0:
            return {
                'neighbor_idx': None,
                'neighbor_frame': None,
                'query_frame': query_frame,
                'query_label': query_label,
                'distance': np.inf,
                'position': query_pos,
                'neighbor_position': None
            }
        
        # Compute distances to all positions with same label and past frames
        past_positions = all_positions[past_frame_indices]
        distances = np.linalg.norm(past_positions - query_pos, axis=1)
        
        # Find nearest neighbor
        nearest_idx_in_subset = np.argmin(distances)
        nearest_idx = past_frame_indices[nearest_idx_in_subset]
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

    def compute_all_nearest_neighbors_different_frame(self, lower_bound_idx=50) -> dict:
        """
        Compute nearest neighbor with same label but different frame ID for all positions.
        
        Args:
            lower_bound_idx: Minimum index to start searching from (warmup window)
        
        Returns:
            Dictionary with structure:
            {
                'query_indices': [indices of query positions],
                'neighbor_indices': [indices of nearest neighbors],
                'labels': [labels of query positions],
                'distances': [distances to nearest neighbors],
                'details': [full result dictionaries]
            }
        """
        query_indices = []
        neighbor_indices = []
        labels = []
        distances = []
        details = []
        
        for i in range(lower_bound_idx, len(self.df)):
            result = self.compute_nearest_neighbor_different_frame(i, lower_bound_idx=lower_bound_idx)
            
            query_indices.append(i)
            neighbor_indices.append(result['neighbor_idx'])
            labels.append(result['query_label'])
            distances.append(result['distance'])
            details.append(result)
        
        return {
            'query_indices': np.array(query_indices),
            'neighbor_indices': np.array(neighbor_indices, dtype=object),  # Can be None
            'labels': np.array(labels),
            'distances': np.array(distances),
            'details': details,
            'lower_bound_idx': lower_bound_idx,
            'total_queries': len(query_indices),
            'valid_neighbors': np.sum([n is not None for n in neighbor_indices])
        }
    
    def get_ground_truth_loop_closure(self, warm_up=100, lower_bound_idx=20, distance_threshold=2.0, topk=None) -> dict:
        """
        Generate ground truth loop closure pairs based on spatial proximity.
        Identifies all pairs where positions have same label but are spatially close.
        
        Retrieval always searches for candidates within PAST frames (lower frame IDs) relative to query.
        
        Args:
            warm_up: Number of initial frames to skip (first n frames are ignored)
            lower_bound_idx: Minimum frame gap to ignore immediate past frames (avoids same-trajectory matches)
            distance_threshold: Maximum distance to consider as loop closure (hard constraint, always applied)
            topk: If specified, keep only top-k closest neighbors per query from those within distance_threshold
            
        Returns:
            Dictionary with ground truth loop closure information:
            {
                'query_indices': [indices of query positions],
                'neighbor_indices': [indices of loop closure neighbors],
                'distances': [distances between pairs],
                'labels': [labels of pairs],
                'statistics': aggregated distance statistics,
                'query_to_neighbor': mapping of query to neighbor info,
                'total_query_positions': number of unique query positions,
                'valid_loop_closures': total number of loop closures found,
                'parameters': configuration used,
                'topk': topk value used
            }
            
        Note:
            Selection uses AND logic: candidates must be within distance_threshold AND 
            (if topk is set) the top-k closest of those candidates are selected.
            This ensures distance_threshold always acts as a hard upper bound.
        """
        query_indices = []
        neighbor_indices = []
        distances_list = []
        labels_list = []
        query_to_neighbor = {}
        
        all_positions = self._get_positions_()
        all_labels = self._get_labels()
        all_frame_ids = self._get_frame_ids()
        
        # Process each position starting from warm_up
        for i in range(warm_up, len(self.df)):
            query_pos = all_positions[i]
            query_label = all_labels[i]
            query_frame = all_frame_ids[i]
            
            # Find eligible indices: past frames only
            # Must be BEFORE current index AND with frame_id at least lower_bound_idx frames in the past
            eligible_mask = (all_frame_ids < query_frame - lower_bound_idx)
            eligible_indices = np.where(eligible_mask)[0]
            
            if len(eligible_indices) == 0:
                continue
            
            # Find positions with SAME label from eligible set
            same_label_mask = all_labels[eligible_indices] == query_label
            same_label_indices = eligible_indices[same_label_mask]
            
            if len(same_label_indices) == 0:
                continue
            
            # Compute distances to all same-label neighbors
            same_label_positions = all_positions[same_label_indices]
            dists = np.linalg.norm(same_label_positions - query_pos, axis=1)
            
            # Apply selection criteria: topk AND distance threshold
            # First filter by distance threshold
            close_mask = dists < distance_threshold
            candidates_indices = same_label_indices[close_mask]
            candidates_distances = dists[close_mask]
            
            if len(candidates_indices) == 0:
                continue
            
            # Then apply topk if specified
            if topk is not None:
                # Select top-k closest among candidates within threshold
                topk_indices = np.argsort(candidates_distances)[:min(topk, len(candidates_distances))]
                selected_indices = candidates_indices[topk_indices]
                selected_distances = candidates_distances[topk_indices]
            else:
                # Keep all candidates within distance threshold
                selected_indices = candidates_indices
                selected_distances = candidates_distances
            
            # Add to results
            for neighbor_idx, neighbor_dist in zip(selected_indices, selected_distances):
                query_indices.append(i)
                neighbor_indices.append(neighbor_idx)
                distances_list.append(neighbor_dist)
                labels_list.append(query_label)
                
                # Store query to neighbor mapping
                if i not in query_to_neighbor:
                    query_to_neighbor[i] = []
                query_to_neighbor[i].append({
                    'neighbor_idx': int(neighbor_idx),
                    'distance': float(neighbor_dist),
                    'label': int(query_label)
                })
        
        # Compute statistics
        distances_array = np.array(distances_list)
        if len(distances_array) > 0:
            statistics = {
                'min_distance': float(np.min(distances_array)),
                'max_distance': float(np.max(distances_array)),
                'mean_distance': float(np.mean(distances_array)),
                'median_distance': float(np.median(distances_array)),
                'std_distance': float(np.std(distances_array))
            }
        else:
            statistics = {
                'min_distance': 0.0,
                'max_distance': 0.0,
                'mean_distance': 0.0,
                'median_distance': 0.0,
                'std_distance': 0.0
            }
        
        return {
            'query_indices': np.array(query_indices),
            'neighbor_indices': np.array(neighbor_indices),
            'distances': np.array(distances_list),
            'labels': np.array(labels_list),
            'statistics': statistics,
            'query_to_neighbor': query_to_neighbor,
            'total_query_positions': len(np.unique(query_indices)),
            'valid_loop_closures': len(query_indices),
            'parameters': {
                'warm_up': warm_up,
                'lower_bound_idx': lower_bound_idx,
                'distance_threshold': distance_threshold,
                'topk': topk
            },
            'topk': topk
        }
    
    def get_nearest_neighbor_ground_truth(self, lower_bound_idx=50) -> dict:
        """
        Get a cleaner representation of ground truth nearest neighbors.
        
        Returns:
            Dictionary with simplified structure for evaluation:
            {
                'query_to_neighbor': {query_idx: neighbor_idx, ...},
                'query_indices': sorted list of query indices,
                'neighbor_indices': sorted list of neighbor indices,
                'statistics': {...}
            }
        """
        nn_data = self.compute_all_nearest_neighbors_different_frame(lower_bound_idx)
        
        query_to_neighbor = {}
        valid_pairs = []
        
        for q_idx, n_idx, dist in zip(
            nn_data['query_indices'],
            nn_data['neighbor_indices'],
            nn_data['distances']
        ):
            if n_idx is not None:
                query_to_neighbor[int(q_idx)] = {
                    'neighbor_idx': int(n_idx),
                    'distance': float(dist),
                    'label': int(nn_data['labels'][len(valid_pairs)])
                }
                valid_pairs.append((int(q_idx), int(n_idx)))
        
        return {
            'query_to_neighbor': query_to_neighbor,
            'query_indices': sorted(query_to_neighbor.keys()),
            'neighbor_indices': sorted(set([v['neighbor_idx'] for v in query_to_neighbor.values()])),
            'total_query_positions': nn_data['total_queries'],
            'valid_loop_closures': len(valid_pairs),
            'statistics': {
                'min_distance': float(np.min(nn_data['distances'][nn_data['distances'] != np.inf])) if np.any(nn_data['distances'] != np.inf) else np.inf,
                'max_distance': float(np.max(nn_data['distances'][nn_data['distances'] != np.inf])) if np.any(nn_data['distances'] != np.inf) else np.inf,
                'mean_distance': float(np.mean(nn_data['distances'][nn_data['distances'] != np.inf])) if np.any(nn_data['distances'] != np.inf) else np.inf,
                'median_distance': float(np.median(nn_data['distances'][nn_data['distances'] != np.inf])) if np.any(nn_data['distances'] != np.inf) else np.inf,
                'lower_bound_idx': lower_bound_idx
            }
        }
    
    def save_ground_truth_loop_closures(self, output_dir=None, warm_up=100, lower_bound_idx=20, distance_threshold=2.0, topk=None):
        """
        Save all ground truth loop closure data to disk in multiple formats.
        
        Saves the following files:
        - ground_truth_loop_closures.csv: Complete table of all loop closure pairs
        - ground_truth_loop_closures.npz: Numpy arrays (query_indices, neighbor_indices, distances, labels)
        - ground_truth_statistics.txt: Summary statistics and metadata
        - ground_truth_parameters.json: Configuration parameters used
        - ground_truth_query_to_neighbor.json: Query to neighbor mapping
        - ground_truth_neighbor_pairs.txt: Human-readable neighbor pairs
        
        Args:
            output_dir: Output directory (default: dataset directory)
            warm_up: Number of initial frames to skip
            lower_bound_idx: Minimum frame gap to ignore immediate past frames
            distance_threshold: Maximum distance for loop closure
            topk: If specified, save only top-k closest neighbors per query
            
        Returns:
            Dictionary with paths to saved files
        """
        import json
        
        # Set output directory
        if output_dir is None:
            output_dir = self.target_dir
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Compute ground truth loop closures with all parameters
        gt_lc = self.get_ground_truth_loop_closure(
            warm_up=warm_up,
            lower_bound_idx=lower_bound_idx,
            distance_threshold=distance_threshold,
            topk=topk
        )
        
        saved_files = {}
        
        # 1. Save CSV with all loop closure pairs
        csv_path = os.path.join(output_dir, 'ground_truth_loop_closures.csv')
        csv_data = pd.DataFrame({
            'query_idx': gt_lc['query_indices'],
            'neighbor_idx': gt_lc['neighbor_indices'],
            'distance_m': gt_lc['distances'],
            'label': gt_lc['labels']
        })
        csv_data.to_csv(csv_path, index=False)
        saved_files['csv'] = csv_path
        
        # 2. Save binary numpy arrays
        npz_path = os.path.join(output_dir, 'ground_truth_loop_closures.npz')
        np.savez(
            npz_path,
            query_indices=gt_lc['query_indices'],
            neighbor_indices=gt_lc['neighbor_indices'],
            distances=gt_lc['distances'],
            labels=gt_lc['labels']
        )
        saved_files['npz'] = npz_path
        
        # 3. Save parameters configuration
        params_path = os.path.join(output_dir, 'ground_truth_parameters.json')
        params_data = {
            'dataset_info': {
                'dataset_root': self.target_dir,
                'sequence': getattr(self, 'sequence', 'unknown')
            },
            'computation_parameters': {
                'warm_up': int(warm_up),
                'lower_bound_idx': int(lower_bound_idx),
                'distance_threshold': float(distance_threshold),
                'topk': int(topk) if topk is not None else None
            },
            'dataset_statistics': {
                'total_positions': len(self.df),
                'total_frames': int(len(np.unique(self._get_frame_ids()))),
                'unique_labels': sorted(np.unique(self._get_labels()).tolist()),
                'label_distribution': {
                    int(label): int(count) 
                    for label, count in zip(*np.unique(self._get_labels(), return_counts=True))
                }
            },
            'results': {
                'total_query_positions': int(gt_lc['total_query_positions']),
                'valid_loop_closures': int(gt_lc['valid_loop_closures']),
                'query_index_range': [int(min(gt_lc['query_indices'])), int(max(gt_lc['query_indices']))],
                'neighbor_index_range': [int(min(gt_lc['neighbor_indices'])), int(max(gt_lc['neighbor_indices']))]
            }
        }
        
        with open(params_path, 'w') as f:
            json.dump(params_data, f, indent=2)
        saved_files['parameters'] = params_path
        
        # 4. Save statistics and metadata
        stats_path = os.path.join(output_dir, 'ground_truth_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("GROUND TRUTH LOOP CLOSURE STATISTICS\n")
            f.write("=" * 90 + "\n\n")
            
            f.write(f"Dataset: {self.target_dir}\n")
            f.write(f"Sequence: {getattr(self, 'sequence', 'unknown')}\n\n")
            
            f.write("CONFIGURATION PARAMETERS:\n")
            f.write(f"  Warm-up frames (skip first N): {warm_up}\n")
            f.write(f"  Lower bound frame gap (ignore immediate past): {lower_bound_idx}\n")
            f.write(f"  Distance threshold: {distance_threshold} m\n")
            f.write(f"  Top-K closest neighbors: {topk if topk is not None else 'All (threshold-based)'}\n\n")
            
            positions = self._get_positions_()
            labels = self._get_labels()
            frame_ids = self._get_frame_ids()
            
            f.write("DATASET STATISTICS:\n")
            f.write(f"  Total positions: {len(positions)}\n")
            f.write(f"  Total frames: {len(np.unique(frame_ids))}\n")
            f.write(f"  Unique labels: {sorted(np.unique(labels).tolist())}\n")
            f.write(f"  Label distribution:\n")
            for label, count in sorted(zip(*np.unique(labels, return_counts=True))):
                f.write(f"    Label {label}: {count} positions\n")
            f.write("\n")
            
            f.write("LOOP CLOSURE RESULTS:\n")
            f.write(f"  Total query positions: {gt_lc['total_query_positions']}\n")
            f.write(f"  Valid loop closures found: {gt_lc['valid_loop_closures']}\n")
            
            if len(gt_lc['query_indices']) > 0:
                f.write(f"  Query indices range: [{min(gt_lc['query_indices'])}, {max(gt_lc['query_indices'])}]\n")
                f.write(f"  Neighbor indices range: [{min(gt_lc['neighbor_indices'])}, {max(gt_lc['neighbor_indices'])}]\n")
            f.write("\n")
            
            f.write("DISTANCE STATISTICS:\n")
            for key, value in gt_lc['statistics'].items():
                f.write(f"  {key}: {value:.6f} m\n")
            f.write("\n")
        
        saved_files['stats'] = stats_path
        
        # 5. Save query to neighbor mapping (JSON format)
        mapping_path = os.path.join(output_dir, 'ground_truth_query_to_neighbor.json')
        mapping_data = {
            'metadata': {
                'dataset': self.target_dir,
                'sequence': getattr(self, 'sequence', 'unknown'),
                'warm_up': int(warm_up),
                'lower_bound_idx': int(lower_bound_idx),
                'distance_threshold': float(distance_threshold),
                'topk': int(topk) if topk is not None else None,
                'total_queries': int(gt_lc['total_query_positions']),
                'total_loop_closures': int(gt_lc['valid_loop_closures'])
            },
            'query_to_neighbor': {
                str(k): v
                for k, v in gt_lc['query_to_neighbor'].items()
            },
            'statistics': {
                k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                for k, v in gt_lc['statistics'].items()
            }
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        saved_files['json'] = mapping_path
        
        # 6. Save human-readable neighbor pairs text file
        txt_path = os.path.join(output_dir, 'ground_truth_neighbor_pairs.txt')
        with open(txt_path, 'w') as f:
            f.write("Ground Truth Loop Closure Neighbor Pairs\n")
            f.write("=" * 90 + "\n")
            f.write(f"Config: warm_up={warm_up}, lower_bound_idx={lower_bound_idx}, ")
            f.write(f"distance_threshold={distance_threshold}, topk={topk}\n\n")
            f.write(f"{'Query':<12} {'Neighbor':<12} {'Distance (m)':<18} {'Label':<10}\n")
            f.write("-" * 90 + "\n")
            
            for query_idx, neighbor_idx, distance, label in zip(
                gt_lc['query_indices'],
                gt_lc['neighbor_indices'],
                gt_lc['distances'],
                gt_lc['labels']
            ):
                f.write(f"{query_idx:<12} {neighbor_idx:<12} {distance:<18.6f} {label:<10}\n")
        
        saved_files['txt'] = txt_path
        
        # Print summary
        print(f"\n{'='*90}")
        print(f"Ground Truth Loop Closures Saved Successfully")
        print(f"{'='*90}")
        print(f"Output directory: {output_dir}\n")
        
        print(f"Configuration Used:")
        print(f"  Warm-up frames: {warm_up}")
        print(f"  Lower bound frame gap: {lower_bound_idx}")
        print(f"  Distance threshold: {distance_threshold} m")
        print(f"  Top-K neighbors: {topk if topk is not None else 'All (threshold-based)'}\n")
        
        print(f"Saved Files:")
        print(f"  CSV table: {os.path.basename(csv_path)}")
        print(f"  NumPy arrays: {os.path.basename(npz_path)}")
        print(f"  Parameters: {os.path.basename(params_path)}")
        print(f"  Statistics: {os.path.basename(stats_path)}")
        print(f"  JSON mapping: {os.path.basename(mapping_path)}")
        print(f"  Text pairs: {os.path.basename(txt_path)}\n")
        
        print(f"Results Summary:")
        print(f"  Total loop closures: {gt_lc['valid_loop_closures']}")
        print(f"  Total query positions: {gt_lc['total_query_positions']}")
        if len(gt_lc['distances']) > 0:
            print(f"  Mean distance: {gt_lc['statistics']['mean_distance']:.4f} m")
            print(f"  Median distance: {gt_lc['statistics']['median_distance']:.4f} m")
            print(f"  Min distance: {gt_lc['statistics']['min_distance']:.4f} m")
            print(f"  Max distance: {gt_lc['statistics']['max_distance']:.4f} m")
        print(f"{'='*90}\n")
        
        return saved_files
