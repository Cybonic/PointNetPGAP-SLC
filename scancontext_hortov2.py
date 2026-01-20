#!/usr/bin/env python3
"""
ScanContext Place Recognition for Hortov2 Dataset

This script generates place recognition predictions using ScanContext algorithm
similar to the PointNetGAP approach but using handcrafted descriptors.

Author: GitHub Copilot
Date: 2026-01-20
"""

import os
import sys
import yaml
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Add scancontext_cpp directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scancontext_cpp'))

# Try to import C++ accelerated version
USE_CPP = False
try:
    import scancontext_cpp
    # Check if the module actually has the functions
    if hasattr(scancontext_cpp, 'distance_sc'):
        USE_CPP = True
        print("[INFO] Using C++ accelerated ScanContext (much faster!)")
    else:
        print("[INFO] C++ module found but incomplete, using Python implementation")
except ImportError:
    print("[INFO] C++ module not found, using Python implementation")
    print("[INFO] To build C++: cd scancontext_cpp && ./build.sh")

# Import or define distance_sc function
if not USE_CPP:
    try:
        from Distance_SC import distance_sc
    except ImportError:
        # If import fails, define it locally
        def distance_sc(sc1, sc2):
            """Compute distance between two ScanContext descriptors."""
            num_sectors = sc1.shape[1]
            sim_for_each_cols = np.zeros(num_sectors)

            for i in range(num_sectors):
                # Shift
                one_step = 1
                sc1 = np.roll(sc1, one_step, axis=1)

                # Compare
                sum_of_cos_sim = 0
                num_col_engaged = 0

                for j in range(num_sectors):
                    col_j_1 = sc1[:, j]
                    col_j_2 = sc2[:, j]

                    if (~np.any(col_j_1) or ~np.any(col_j_2)):
                        continue

                    # Calc sim
                    cos_similarity = np.dot(col_j_1, col_j_2) / (np.linalg.norm(col_j_1) * np.linalg.norm(col_j_2))
                    sum_of_cos_sim = sum_of_cos_sim + cos_similarity
                    num_col_engaged = num_col_engaged + 1

                sim_for_each_cols[i] = sum_of_cos_sim / num_col_engaged

            sim = max(sim_for_each_cols)
            dist = 1 - sim
            return dist
else:
    # Use C++ implementation
    distance_sc = scancontext_cpp.distance_sc


class ScanContextGenerator:
    """
    Generate ScanContext descriptors from point clouds.
    
    Based on the original ScanContext implementation adapted for numpy arrays.
    """
    
    def __init__(self, sector_res=60, ring_res=20, max_length=80, 
                 downcell_size=0.5, lidar_height=0.0):
        """
        Initialize ScanContext generator.
        
        Args:
            sector_res: Number of sectors (azimuth bins)
            ring_res: Number of rings (radial bins)
            max_length: Maximum distance to consider (meters)
            downcell_size: Voxel size for downsampling
            lidar_height: Height offset of the lidar sensor
        """
        self.sector_res = sector_res
        self.ring_res = ring_res
        self.max_length = max_length
        self.downcell_size = downcell_size
        self.lidar_height = lidar_height
        
    def xy2theta(self, x, y):
        """Convert XY coordinates to angle in degrees."""
        if x >= 0 and y >= 0:
            theta = 180 / np.pi * np.arctan(y / (x + 1e-6))
        elif x < 0 and y >= 0:
            theta = 180 - (180 / np.pi * np.arctan(y / (-x + 1e-6)))
        elif x < 0 and y < 0:
            theta = 180 + (180 / np.pi * np.arctan(y / x))
        elif x >= 0 and y < 0:
            theta = 360 - (180 / np.pi * np.arctan(-y / (x + 1e-6)))
        return theta
    
    def pt2rs(self, point, gap_ring, gap_sector, num_ring, num_sector):
        """Convert point to ring and sector indices."""
        x, y, z = point[0], point[1], point[2]
        
        if x == 0.0:
            x = 0.001
        if y == 0.0:
            y = 0.001
        
        theta = self.xy2theta(x, y)
        faraway = np.sqrt(x*x + y*y)
        
        idx_ring = int(faraway / gap_ring)
        idx_sector = int(theta / gap_sector)
        
        if idx_ring >= num_ring:
            idx_ring = num_ring - 1
            
        return idx_ring, idx_sector
    
    def downsample_voxel(self, points, voxel_size):
        """Simple voxel downsampling."""
        if voxel_size <= 0:
            return points
            
        # Compute voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(np.int32)
        
        # Use pandas to get unique voxels and their first occurrence
        df = pd.DataFrame(voxel_indices, columns=['x', 'y', 'z'])
        df['idx'] = np.arange(len(df))
        unique_voxels = df.drop_duplicates(subset=['x', 'y', 'z'], keep='first')
        
        return points[unique_voxels['idx'].values]
    
    def ptcloud2sc(self, ptcloud):
        """
        Convert point cloud to ScanContext descriptor.
        
        Args:
            ptcloud: Numpy array of shape (N, 3) or (N, 4)
            
        Returns:
            ScanContext descriptor of shape (ring_res, sector_res)
        """
        # Extract XYZ if input has intensity
        if ptcloud.shape[1] > 3:
            ptcloud = ptcloud[:, :3]
        
        num_points = ptcloud.shape[0]
        gap_ring = self.max_length / self.ring_res
        gap_sector = 360 / self.sector_res
        
        enough_large = 1000
        sc_storage = np.zeros([enough_large, self.ring_res, self.sector_res])
        sc_counter = np.zeros([self.ring_res, self.sector_res])
        
        for pt_idx in range(num_points):
            point = ptcloud[pt_idx, :]
            point_height = point[2] + self.lidar_height
            
            idx_ring, idx_sector = self.pt2rs(point, gap_ring, gap_sector, 
                                              self.ring_res, self.sector_res)
            
            if sc_counter[idx_ring, idx_sector] >= enough_large:
                continue
                
            sc_storage[int(sc_counter[idx_ring, idx_sector]), idx_ring, idx_sector] = point_height
            sc_counter[idx_ring, idx_sector] += 1
        
        # Take maximum height in each bin
        sc = np.amax(sc_storage, axis=0)
        
        return sc
    
    def generate(self, ptcloud):
        """
        Generate ScanContext descriptor from point cloud.
        
        Args:
            ptcloud: Numpy array of shape (N, 3) or (N, 4)
            
        Returns:
            ScanContext descriptor
        """
        # Downsample
        if self.downcell_size > 0:
            ptcloud = self.downsample_voxel(ptcloud, self.downcell_size)
        
        # Generate ScanContext
        sc = self.ptcloud2sc(ptcloud)
        
        return sc


class HortoV2Dataset:
    """
    Handler for HortoV2 dataset structure.
    """
    
    def __init__(self, dataset_root, sequence):
        """
        Initialize dataset handler.
        
        Args:
            dataset_root: Root directory of the dataset
            sequence: Sequence name (e.g., 'PCD_EASY', 'PCD_Easy_DARK')
        """
        self.dataset_root = Path(dataset_root)
        self.sequence = sequence
        self.seq_path = self.dataset_root / sequence
        
        # Load metadata
        self.load_metadata()
        
    def load_metadata(self):
        """Load sequence metadata (poses, labels, etc.)."""
        # Look for path_easy.csv at sequence root level
        csv_file = self.seq_path / 'path_easy.csv'
        
        if not csv_file.exists():
            # Try to find any CSV file
            csv_files = list(self.seq_path.glob('*.csv'))
            if not csv_files:
                # Try in subdirectories
                csv_files = list(self.seq_path.glob('**/*.csv'))
            
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.seq_path}")
            
            csv_file = csv_files[0]
        
        self.csv_path = csv_file
        print(f"[INFO] Loading metadata from: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path, comment='/')
        self.df.columns = self.df.columns.str.strip().str.replace('"', '')
        
        # Get positions
        self.positions = self.df[['x', 'y', 'z']].values
        
        # Get labels if available
        if 'label' in self.df.columns:
            self.labels = self.df['label'].values
        else:
            self.labels = np.zeros(len(self.df))
        
        # Get point cloud directory
        # Try common names: pcd, pointcloud, velodyne
        pcd_dir_names = ['pcd', 'pointcloud', 'velodyne', 'lidar']
        self.pcd_dir = None
        
        for dir_name in pcd_dir_names:
            candidate = self.seq_path / dir_name
            if candidate.exists() and candidate.is_dir():
                self.pcd_dir = candidate
                break
        
        if self.pcd_dir is None:
            # Try to find any directory with point cloud files
            for subdir in self.seq_path.iterdir():
                if subdir.is_dir():
                    pcd_files = list(subdir.glob('*.pcd')) + list(subdir.glob('*.bin'))
                    if pcd_files:
                        self.pcd_dir = subdir
                        break
        
        if self.pcd_dir is None:
            raise FileNotFoundError(f"No pointcloud directory found in {self.seq_path}")
        
        print(f"[INFO] Point cloud directory: {self.pcd_dir}")
        
        # Get list of point cloud files
        self.pcd_files = sorted(list(self.pcd_dir.glob('*.bin')) + 
                               list(self.pcd_dir.glob('*.pcd')))
        
        print(f"[INFO] Found {len(self.pcd_files)} point cloud files")
        print(f"[INFO] Metadata rows: {len(self.df)}")
        
    def load_pointcloud(self, idx):
        """
        Load point cloud at given index.
        
        Args:
            idx: Index of the point cloud
            
        Returns:
            Numpy array of shape (N, 3) or (N, 4)
        """
        if idx >= len(self.pcd_files):
            raise IndexError(f"Index {idx} out of range [0, {len(self.pcd_files)})")
        
        pcd_file = self.pcd_files[idx]
        
        # Load based on file extension
        if pcd_file.suffix == '.bin':
            # KITTI format: float32 x, y, z, intensity
            points = np.fromfile(str(pcd_file), dtype=np.float32)
            points = points.reshape((-1, 4))
        elif pcd_file.suffix == '.pcd':
            # Load PCD file (simplified - assumes ASCII format)
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(pcd_file))
            points = np.asarray(pcd.points)
        else:
            raise ValueError(f"Unsupported file format: {pcd_file.suffix}")
        
        return points
    
    def __len__(self):
        """Return number of samples."""
        return len(self.pcd_files)


class ScanContextPlaceRecognition:
    """
    Place recognition using ScanContext descriptors.
    """
    
    def __init__(self, dataset, sc_generator, logger=None):
        """
        Initialize place recognition system.
        
        Args:
            dataset: Dataset handler
            sc_generator: ScanContext generator
            logger: Logger instance
        """
        self.dataset = dataset
        self.sc_generator = sc_generator
        self.logger = logger or self._create_logger()
        
        self.descriptors = {}
        
    def _create_logger(self):
        """Create default logger."""
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def generate_descriptors(self):
        """
        Generate ScanContext descriptors for all point clouds.
        
        Returns:
            Dictionary mapping indices to descriptors
        """
        self.logger.info(f"Generating ScanContext descriptors for {len(self.dataset)} samples...")
        
        for idx in tqdm(range(len(self.dataset)), desc="Generating ScanContext"):
            try:
                # Load point cloud
                points = self.dataset.load_pointcloud(idx)
                
                # Generate descriptor
                sc = self.sc_generator.generate(points)
                
                # Store descriptor
                self.descriptors[idx] = {
                    'd': sc.flatten().tolist(),  # Flatten for compatibility
                    'sc': sc  # Keep 2D version for distance computation
                }
                
            except Exception as e:
                self.logger.error(f"Failed to process sample {idx}: {e}")
                # Use zero descriptor as fallback
                self.descriptors[idx] = {
                    'd': np.zeros(self.sc_generator.ring_res * self.sc_generator.sector_res).tolist(),
                    'sc': np.zeros((self.sc_generator.ring_res, self.sc_generator.sector_res))
                }
        
        return self.descriptors
    
    def compute_predictions(self, top_k=25, warmup_window=100, roi_window=50):
        """
        Compute top-k predictions for each query.
        
        Args:
            top_k: Number of top candidates to retrieve
            warmup_window: Number of initial frames to skip
            roi_window: Window size to exclude nearby frames
            
        Returns:
            Dictionary with predictions
        """
        n_samples = len(self.descriptors)
        
        # Use C++ accelerated version if available
        if USE_CPP:
            self.logger.info(f"Computing predictions (top-{top_k}) using C++ acceleration...")
            
            # Prepare data for C++ function
            all_scs = [self.descriptors[i]['sc'] for i in range(n_samples)]
            query_indices = list(range(warmup_window, n_samples))
            
            # Call C++ function
            predictions = scancontext_cpp.compute_top_k_predictions(
                all_scs, query_indices, roi_window, top_k
            )
            
            # Convert to expected format
            result = {}
            for query_idx, top_k_indices in predictions.items():
                # Compute distances for the top-k (for compatibility)
                distances = [
                    distance_sc(self.descriptors[query_idx]['sc'], 
                               self.descriptors[idx]['sc'])
                    for idx in top_k_indices
                ]
                result[query_idx] = {
                    'top_k_indices': top_k_indices,
                    'top_k_distances': distances
                }
            
            return result
        
        # Python fallback implementation
        predictions = {}
        
        self.logger.info(f"Computing predictions (top-{top_k}) using Python...")
        
        for query_idx in tqdm(range(warmup_window, n_samples), desc="Computing predictions"):
            # Compute distances to all database samples
            distances = []
            
            query_sc = self.descriptors[query_idx]['sc']
            
            # Only consider samples outside ROI window
            for db_idx in range(query_idx - roi_window):
                db_sc = self.descriptors[db_idx]['sc']
                
                # Compute ScanContext distance
                dist = distance_sc(query_sc, db_sc)
                distances.append((dist, db_idx))
            
            # Sort by distance (ascending)
            distances.sort(key=lambda x: x[0])
            
            # Get top-k predictions
            top_k_indices = [idx for _, idx in distances[:top_k]]
            top_k_distances = [dist for dist, _ in distances[:top_k]]
            
            predictions[query_idx] = {
                'top_k_indices': top_k_indices,
                'top_k_distances': top_k_distances
            }
        
        return predictions
    
    def load_descriptors(self, load_path):
        """
        Load descriptors from file.
        
        Args:
            load_path: Path to the descriptors file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(load_path):
            return False
        
        try:
            with open(load_path, 'rb') as f:
                loaded_desc = pickle.load(f)
            
            # Convert back to internal format with both 'd' and 'sc'
            self.descriptors = {}
            for idx, desc in loaded_desc.items():
                flat_desc = desc['d']
                # Reshape to 2D for distance computation
                sc_2d = np.array(flat_desc).reshape(self.sc_generator.ring_res, 
                                                     self.sc_generator.sector_res)
                self.descriptors[idx] = {
                    'd': flat_desc,
                    'sc': sc_2d
                }
            
            self.logger.info(f"Loaded {len(self.descriptors)} descriptors from: {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load descriptors: {e}")
            return False
    
    def save_descriptors(self, save_path):
        """Save descriptors to file."""
        # Convert to format compatible with PointNetGAP
        descriptors_to_save = {
            idx: {'d': desc['d']} 
            for idx, desc in self.descriptors.items()
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(descriptors_to_save, f)
        
        self.logger.info(f"Saved descriptors to: {save_path}")
    
    def save_predictions(self, predictions, save_path):
        """Save predictions to file."""
        with open(save_path, 'wb') as f:
            pickle.dump(predictions, f)
        
        self.logger.info(f"Saved predictions to: {save_path}")
    
    def compute_recall(self, predictions, distance_thresholds=[5, 10, 15, 20], top_k_list=[1, 5, 10, 25]):
        """
        Compute recall metrics at different distance thresholds and top-k values.
        Computes both global recall and per-segment (row) recall.
        
        Args:
            predictions: Dictionary with predictions
            distance_thresholds: List of distance thresholds in meters
            top_k_list: List of top-k values to evaluate
            
        Returns:
            Dictionary with recall metrics including global and per-segment
        """
        self.logger.info(f"Computing recall metrics...")
        
        positions = self.dataset.positions
        labels = self.dataset.labels
        n_samples = len(positions)
        
        # Get unique segments/rows
        unique_labels = sorted(set(labels))
        self.logger.info(f"Found {len(unique_labels)} unique row segments: {unique_labels}")
        
        # Initialize metrics - add segments
        recall_results = {
            'global': {f'{dist}m': {f'top{k}': [] for k in top_k_list} for dist in distance_thresholds},
            'segments': {label: {f'{dist}m': {f'top{k}': [] for k in top_k_list} for dist in distance_thresholds} 
                        for label in unique_labels},
            'per_query': {}
        }
        
        # For each query
        for query_idx in tqdm(predictions.keys(), desc="Computing recall"):
            query_pos = positions[query_idx]
            top_k_indices = predictions[query_idx]['top_k_indices']
            
            # Compute ground truth: all frames within distance threshold (excluding ROI window)
            gt_distances = np.linalg.norm(positions[:query_idx - 50] - query_pos, axis=1)
            
            recall_results['per_query'][query_idx] = {}
            
            # For each distance threshold
            for dist_thresh in distance_thresholds:
                gt_positives = np.where(gt_distances <= dist_thresh)[0]
                
                recall_results['per_query'][query_idx][f'{dist_thresh}m'] = {}
                
                # For each top-k
                for k in top_k_list:
                    # Get top-k predictions
                    pred_indices = top_k_indices[:k]
                    
                    # Check if any prediction is a true positive
                    is_correct = len(np.intersect1d(pred_indices, gt_positives)) > 0
                    
                    recall_results['per_query'][query_idx][f'{dist_thresh}m'][f'top{k}'] = 1 if is_correct else 0
        
        # Aggregate global recall
        for dist_thresh in distance_thresholds:
            for k in top_k_list:
                recalls = [recall_results['per_query'][q][f'{dist_thresh}m'][f'top{k}'] 
                          for q in predictions.keys()]
                recall_results['global'][f'{dist_thresh}m'][f'top{k}'] = np.mean(recalls)
        
        # Aggregate per-segment recall
        for label in unique_labels:
            # Get queries from this segment
            segment_queries = [q for q in predictions.keys() if labels[q] == label]
            
            if len(segment_queries) == 0:
                self.logger.warning(f"No queries found for segment {label}")
                continue
            
            for dist_thresh in distance_thresholds:
                for k in top_k_list:
                    recalls = [recall_results['per_query'][q][f'{dist_thresh}m'][f'top{k}'] 
                              for q in segment_queries]
                    recall_results['segments'][label][f'{dist_thresh}m'][f'top{k}'] = np.mean(recalls)
        
        return recall_results
    
    def save_recall_csv(self, recall_results, save_dir):
        """
        Save recall results to CSV files in PointNetGAP-compatible format.
        Saves both global recall and per-segment (row) recall.
        
        The main format is a matrix where:
        - Rows = top-k values (1, 2, ..., 25)
        - Columns = distance thresholds (0, 1, 2, ..., 119 meters)
        - Values = recall at that (top_k, distance) combination
        
        Args:
            recall_results: Dictionary with recall metrics (global + segments)
            save_dir: Directory to save CSV files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract top-k list and distance thresholds
        distance_keys = sorted(recall_results['global'].keys(), key=lambda x: int(x.replace('m', '')))
        distance_thresholds = [int(k.replace('m', '')) for k in distance_keys]
        
        # Get top-k list from the first distance threshold
        top_k_dict = recall_results['global'][distance_keys[0]]
        top_k_list = sorted([int(k.replace('top', '')) for k in top_k_dict.keys()])
        
        # ============================================================
        # Save GLOBAL recall matrix
        # ============================================================
        recall_matrix = []
        for k in top_k_list:
            row = []
            for dist_key in distance_keys:
                k_key = f'top{k}'
                if k_key in recall_results['global'][dist_key]:
                    recall = recall_results['global'][dist_key][k_key]
                    row.append(recall)
                else:
                    row.append(0.0)
            recall_matrix.append(row)
        
        # Create DataFrame with distance thresholds as column names
        df_recall = pd.DataFrame(recall_matrix, columns=distance_thresholds)
        df_recall.index = top_k_list
        
        # Save main recall.csv file (PointNetGAP format)
        csv_path = save_dir / 'recall.csv'
        df_recall.to_csv(csv_path)
        self.logger.info(f"Saved global recall matrix to: {csv_path}")

        
        # ============================================================
        # Save PER-SEGMENT recall matrices
        # ============================================================
        if 'segments' in recall_results:
            unique_labels = sorted(recall_results['segments'].keys())
            self.logger.info(f"Saving per-segment recall for {len(unique_labels)} segments: {unique_labels}")
            
            for label in unique_labels:
                # Create recall matrix for this segment
                segment_matrix = []
                for k in top_k_list:
                    row = []
                    for dist_key in distance_keys:
                        k_key = f'top{k}'
                        if k_key in recall_results['segments'][label][dist_key]:
                            recall = recall_results['segments'][label][dist_key][k_key]
                            row.append(recall)
                        else:
                            row.append(0.0)
                    segment_matrix.append(row)
                
                # Create DataFrame
                df_segment = pd.DataFrame(segment_matrix, columns=distance_thresholds)
                df_segment.index = top_k_list
                
                # Save segment recall matrix
                csv_path = save_dir / f'recall_{label}.csv'
                df_segment.to_csv(csv_path)
                self.logger.info(f"Saved segment {label} recall matrix to: {csv_path}")



def create_argument_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Generate ScanContext descriptors for HortoV2 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='dataset/PlaceRecognitionTestPolyTunnel',
        help='Root directory of the dataset'
    )
    
    parser.add_argument(
        '--sequence',
        type=str,
        default='PCD_EASY',
        help='Sequence name to process'
    )
    
    parser.add_argument(
        '--session',
        type=str,
        default='hortov2',
        help='Session configuration name'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='saved/hortov2',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--sector_res',
        type=int,
        default=60,
        help='Number of sectors (azimuth resolution)'
    )
    
    parser.add_argument(
        '--ring_res',
        type=int,
        default=20,
        help='Number of rings (radial resolution)'
    )
    
    parser.add_argument(
        '--max_length',
        type=int,
        default=80,
        help='Maximum distance to consider (meters)'
    )
    
    parser.add_argument(
        '--top_k',
        type=int,
        default=25,
        help='Number of top candidates to retrieve'
    )
    
    parser.add_argument(
        '--force_regenerate',
        action='store_true',
        help='Force regeneration of descriptors even if they exist'
    )
    
    parser.add_argument(
        '--skip_predictions',
        action='store_true',
        help='Skip prediction computation (only generate/load descriptors)'
    )
    
    parser.add_argument(
        '--distance_thresholds',
        type=int,
        nargs='+',
        default=[5, 10, 15, 20],
        help='Distance thresholds for recall computation (meters)'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = create_argument_parser()
    
    # Setup logging
    log_file = f'logs/scancontext-{args.sequence}.log'
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("ScanContext Place Recognition for HortoV2")
    logger.info("="*80)
    logger.info(f"Dataset root: {args.dataset_root}")
    logger.info(f"Sequence: {args.sequence}")
    logger.info(f"Sector resolution: {args.sector_res}")
    logger.info(f"Ring resolution: {args.ring_res}")
    logger.info(f"Max length: {args.max_length}m")
    logger.info("="*80)
    
    # Initialize components
    try:
        # Load dataset
        dataset = HortoV2Dataset(args.dataset_root, args.sequence)
        
        # Create ScanContext generator
        sc_generator = ScanContextGenerator(
            sector_res=args.sector_res,
            ring_res=args.ring_res,
            max_length=args.max_length
        )
        
        # Create place recognition system
        pr_system = ScanContextPlaceRecognition(dataset, sc_generator, logger)
        
        # Setup output directory
        output_dir = Path(args.output_dir) / args.sequence / 'ScanContext'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        descriptors_file = output_dir / 'descriptors.pkl'
        predictions_file = output_dir / 'predictions.pkl'
        
        # Check if descriptors already exist
        descriptors_loaded = False
        if descriptors_file.exists() and not args.force_regenerate:
            logger.info(f"Found existing descriptors at: {descriptors_file}")
            logger.info("Attempting to load...")
            descriptors_loaded = pr_system.load_descriptors(descriptors_file)
            
            if descriptors_loaded:
                logger.info("✓ Successfully loaded existing descriptors")
                logger.info("Skipping descriptor generation (already computed)")
            else:
                logger.warning("✗ Failed to load descriptors, will regenerate")
        elif descriptors_file.exists() and args.force_regenerate:
            logger.info(f"Found existing descriptors but --force_regenerate flag is set")
            logger.info("Will regenerate descriptors...")
        
        # Generate descriptors only if not loaded
        if not descriptors_loaded:
            logger.info("Generating new descriptors...")
            descriptors = pr_system.generate_descriptors()
            # Save descriptors immediately after generation
            pr_system.save_descriptors(descriptors_file)
        
        # Skip predictions if requested
        if args.skip_predictions:
            logger.info("Skipping prediction computation (--skip_predictions flag set)")
            logger.info("="*80)
            logger.info("Processing complete!")
            logger.info(f"Descriptors saved to: {descriptors_file}")
            logger.info("="*80)
            return
        
        # Check if predictions already exist
        if predictions_file.exists():
            logger.info(f"Found existing predictions at: {predictions_file}")
            logger.info("Predictions will be recomputed")
        
        # Compute predictions
        predictions = pr_system.compute_predictions(top_k=args.top_k)
        
        # Save results
        pr_system.save_predictions(predictions, predictions_file)
        
        # Compute recall metrics
        logger.info("="*80)
        logger.info("Computing recall metrics...")
        logger.info("="*80)
        
        # distance_thresholds = args.distance_thresholds
        distance_thresholds = list(range(0,120,1))
        
        # Compute recall for all top-k from 1 to 25
        top_k_eval = list(range(1, min(args.top_k + 1, 26)))  # 1, 2, 3, ..., 25
        
        # Add 1% of database size as a metric
        num_samples = len(dataset)
        top_1_percent = max(1, int(0.01 * num_samples))  # At least 1
        if top_1_percent not in top_k_eval and top_1_percent <= args.top_k:
            top_k_eval.append(top_1_percent)
            top_k_eval.sort()
        
        logger.info(f"Evaluating recall for top-k: {top_k_eval}")
        logger.info(f"Top 1% corresponds to top-{top_1_percent} candidates")
        
        recall_results = pr_system.compute_recall(
            predictions, 
            distance_thresholds=distance_thresholds,
            top_k_list=top_k_eval
        )
        
        # Save recall results
        pr_system.save_recall_csv(recall_results, output_dir)
        
        # Log recall results
        logger.info("="*80)
        logger.info("Recall Results:")
        logger.info("="*80)
        for dist_key in recall_results['global'].keys():
            logger.info(f"\nDistance threshold: {dist_key}")
            for k_key, recall_val in recall_results['global'][dist_key].items():
                logger.info(f"  Recall@{k_key}: {recall_val:.4f}")
        
        # Save parameters
        params = {
            'sequence': args.sequence,
            'sector_res': args.sector_res,
            'ring_res': args.ring_res,
            'max_length': args.max_length,
            'top_k': args.top_k,
            'num_samples': len(dataset),
            'method': 'ScanContext',
            'distance_thresholds': distance_thresholds,
            'recall_computed': True
        }
        
        with open(output_dir / 'params.yaml', 'w') as f:
            yaml.dump(params, f)
        
        logger.info("="*80)
        logger.info("Processing complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
