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

# Note: PlaceRecognition is NOT imported at top level to avoid torch dependency
# The compute_recall logic is implemented locally in _compute_recall_from_predictions_impl

# Try to import GPU acceleration (PyTorch)
USE_GPU = True
torch = None
try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        print(f"[INFO] GPU available - using PyTorch CUDA acceleration")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] PyTorch installed but no CUDA GPU available")
except ImportError:
    print("[INFO] PyTorch not installed - GPU acceleration disabled")

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
        # Try to use Numba for acceleration
        USE_NUMBA = False
        try:
            from numba import jit, prange
            USE_NUMBA = True
            print("[INFO] Numba available - using JIT acceleration")
        except ImportError:
            print("[INFO] Numba not available - using vectorized NumPy")
        
        if USE_NUMBA:
            @jit(nopython=True)
            def distance_sc(sc1, sc2):
                """
                Compute distance between two ScanContext descriptors.
                Numba JIT compiled for maximum performance.
                """
                num_sectors = sc1.shape[1]
                num_rings = sc1.shape[0]
                
                best_sim = -1.0
                
                for shift in range(num_sectors):
                    sum_cos_sim = 0.0
                    num_valid = 0
                    
                    for j in range(num_sectors):
                        # Get shifted column index
                        j_shifted = (j + shift) % num_sectors
                        
                        # Compute norms
                        norm1 = 0.0
                        norm2 = 0.0
                        dot_prod = 0.0
                        
                        for i in range(num_rings):
                            v1 = sc1[i, j_shifted]
                            v2 = sc2[i, j]
                            dot_prod += v1 * v2
                            norm1 += v1 * v1
                            norm2 += v2 * v2
                        
                        norm1 = np.sqrt(norm1)
                        norm2 = np.sqrt(norm2)
                        
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            cos_sim = dot_prod / (norm1 * norm2)
                            sum_cos_sim += cos_sim
                            num_valid += 1
                    
                    if num_valid > 0:
                        sim = sum_cos_sim / num_valid
                        if sim > best_sim:
                            best_sim = sim
                
                return 1.0 - best_sim if best_sim > 0 else 1.0
            
            @jit(nopython=True)
            def distance_sc_with_norms(sc1, sc2, sc1_col_norms, sc2_col_norms):
                """
                Compute distance with precomputed column norms.
                Much faster when comparing one query against many database entries.
                """
                num_sectors = sc1.shape[1]
                num_rings = sc1.shape[0]
                
                best_sim = -1.0
                
                for shift in range(num_sectors):
                    sum_cos_sim = 0.0
                    num_valid = 0
                    
                    for j in range(num_sectors):
                        j_shifted = (j + shift) % num_sectors
                        
                        norm1 = sc1_col_norms[j_shifted]
                        norm2 = sc2_col_norms[j]
                        
                        if norm1 > 1e-8 and norm2 > 1e-8:
                            # Compute dot product only
                            dot_prod = 0.0
                            for i in range(num_rings):
                                dot_prod += sc1[i, j_shifted] * sc2[i, j]
                            
                            cos_sim = dot_prod / (norm1 * norm2)
                            sum_cos_sim += cos_sim
                            num_valid += 1
                    
                    if num_valid > 0:
                        sim = sum_cos_sim / num_valid
                        if sim > best_sim:
                            best_sim = sim
                
                return 1.0 - best_sim if best_sim > 0 else 1.0
            
            @jit(nopython=True)
            def compute_column_norms(sc):
                """Precompute column norms for a ScanContext."""
                num_sectors = sc.shape[1]
                num_rings = sc.shape[0]
                norms = np.zeros(num_sectors)
                
                for j in range(num_sectors):
                    sum_sq = 0.0
                    for i in range(num_rings):
                        sum_sq += sc[i, j] * sc[i, j]
                    norms[j] = np.sqrt(sum_sq)
                
                return norms
            
            @jit(nopython=True, parallel=True)
            def batch_distance_sc_numba(query_sc, query_norms, db_stack, db_norms_stack):
                """
                Compute distances from query to all database SCs.
                Parallelized with Numba, using precomputed norms.
                """
                n_db = db_stack.shape[0]
                distances = np.zeros(n_db)
                
                for db_idx in prange(n_db):
                    distances[db_idx] = distance_sc_with_norms(
                        query_sc, db_stack[db_idx], 
                        query_norms, db_norms_stack[db_idx]
                    )
                
                return distances
            
            def batch_distance_sc(query_sc, database_scs, query_norms=None, db_norms_list=None):
                """Wrapper for Numba batch function with optional precomputed norms."""
                db_stack = np.stack(database_scs, axis=0).astype(np.float64)
                query_sc = np.ascontiguousarray(query_sc.astype(np.float64))
                
                if query_norms is None:
                    query_norms = compute_column_norms(query_sc)
                
                if db_norms_list is None:
                    # Compute norms for all database entries
                    db_norms_stack = np.zeros((len(database_scs), query_sc.shape[1]))
                    for i, sc in enumerate(database_scs):
                        db_norms_stack[i] = compute_column_norms(sc.astype(np.float64))
                else:
                    db_norms_stack = np.stack(db_norms_list, axis=0)
                
                return batch_distance_sc_numba(query_sc, query_norms, db_stack, db_norms_stack)
        
        else:
            # Vectorized NumPy fallback (no Numba)
            def distance_sc(sc1, sc2):
                """
                Compute distance between two ScanContext descriptors.
                Vectorized implementation for better performance.
                """
                num_sectors = sc1.shape[1]
                
                # Precompute norms for sc2 columns
                sc2_norms = np.linalg.norm(sc2, axis=0)
                
                best_sim = -1.0
                
                for shift in range(num_sectors):
                    # Shift sc1
                    sc1_shifted = np.roll(sc1, shift, axis=1)
                    
                    # Compute norms for shifted sc1
                    sc1_norms = np.linalg.norm(sc1_shifted, axis=0)
                    
                    # Find valid columns (both non-zero)
                    valid = (sc1_norms > 1e-8) & (sc2_norms > 1e-8)
                    
                    if not np.any(valid):
                        continue
                    
                    # Vectorized dot product for all columns
                    dot_products = np.sum(sc1_shifted * sc2, axis=0)
                    
                    # Cosine similarities for valid columns
                    cos_sims = dot_products[valid] / (sc1_norms[valid] * sc2_norms[valid])
                    
                    # Average similarity for this shift
                    sim = np.mean(cos_sims)
                    
                    if sim > best_sim:
                        best_sim = sim
                
                return 1.0 - best_sim if best_sim > 0 else 1.0
            
            def batch_distance_sc(query_sc, database_scs):
                """
                Compute distances from one query to multiple database descriptors.
                Optimized batch processing.
                
                Args:
                    query_sc: Query ScanContext (ring_res, sector_res)
                    database_scs: List of database ScanContexts
                    
                Returns:
                    Array of distances
                """
                n_db = len(database_scs)
                distances = np.zeros(n_db)
                
                num_sectors = query_sc.shape[1]
                
                # Stack all database SCs for vectorized operations
                db_stack = np.stack(database_scs, axis=0)  # (n_db, ring_res, sector_res)
                
                # Precompute norms for all database SCs
                db_norms = np.linalg.norm(db_stack, axis=1)  # (n_db, sector_res)
                
                best_sims = np.full(n_db, -1.0)
                
                for shift in range(num_sectors):
                    # Shift query
                    query_shifted = np.roll(query_sc, shift, axis=1)
                    query_norms = np.linalg.norm(query_shifted, axis=0)  # (sector_res,)
                    
                    # Valid columns for query
                    query_valid = query_norms > 1e-8
                    
                    # Dot products: (n_db, sector_res)
                    dot_products = np.sum(query_shifted[np.newaxis, :, :] * db_stack, axis=1)
                    
                    # Denominator: (n_db, sector_res)
                    denom = query_norms[np.newaxis, :] * db_norms
                    
                    # Valid mask: both query and db columns non-zero
                    valid_mask = query_valid[np.newaxis, :] & (db_norms > 1e-8)
                    
                    # Compute similarities
                    for i in range(n_db):
                        valid_i = valid_mask[i]
                        if np.any(valid_i):
                            cos_sims = dot_products[i, valid_i] / denom[i, valid_i]
                            sim = np.mean(cos_sims)
                            if sim > best_sims[i]:
                                best_sims[i] = sim
                
                distances = 1.0 - best_sims
                distances[best_sims < 0] = 1.0
                
                return distances
else:
    # Use C++ implementation
    distance_sc = scancontext_cpp.distance_sc


# ============================================================
# GPU-Accelerated Batch Distance (PyTorch CUDA)
# ============================================================
if USE_GPU:
    def batch_distance_sc_gpu(query_sc, database_scs, query_norms=None, db_norms_list=None):
        """
        GPU-accelerated batch ScanContext distance computation using PyTorch.
        
        Computes distances from one query to all database entries using GPU.
        Significantly faster for large databases (100+ entries).
        
        Args:
            query_sc: Query ScanContext (ring_res, sector_res) - numpy array
            database_scs: List of database ScanContexts - list of numpy arrays
            query_norms: Precomputed column norms for query (optional, unused)
            db_norms_list: Precomputed column norms for database (optional, unused)
            
        Returns:
            Array of distances (numpy)
        """
        device = torch.device('cuda')
        
        # Transfer data to GPU
        query_gpu = torch.from_numpy(query_sc.astype(np.float32)).to(device)
        db_stack_gpu = torch.from_numpy(np.stack(database_scs, axis=0).astype(np.float32)).to(device)
        
        n_db = db_stack_gpu.shape[0]
        num_sectors = query_gpu.shape[1]
        
        # Precompute norms on GPU
        db_norms_gpu = torch.linalg.norm(db_stack_gpu, dim=1)  # (n_db, sector_res)
        
        best_sims = torch.full((n_db,), -1.0, dtype=torch.float32, device=device)
        
        for shift in range(num_sectors):
            # Shift query on GPU
            query_shifted = torch.roll(query_gpu, shifts=shift, dims=1)
            query_norms_t = torch.linalg.norm(query_shifted, dim=0)  # (sector_res,)
            
            # Valid columns for query
            query_valid = query_norms_t > 1e-8
            
            # Vectorized dot products for all database entries: (n_db, sector_res)
            dot_products = torch.sum(query_shifted.unsqueeze(0) * db_stack_gpu, dim=1)
            
            # Denominator: (n_db, sector_res)
            denom = query_norms_t.unsqueeze(0) * db_norms_gpu
            
            # Valid mask: both query and db columns non-zero
            valid_mask = query_valid.unsqueeze(0) & (db_norms_gpu > 1e-8)
            
            # Compute cosine similarity where valid
            cos_sims = torch.where(valid_mask & (denom > 1e-8), 
                                   dot_products / (denom + 1e-8), 
                                   torch.zeros_like(dot_products))
            
            # Count valid elements per row
            valid_counts = torch.sum(valid_mask.float(), dim=1)
            
            # Sum of cosine similarities per row
            cos_sums = torch.sum(cos_sims, dim=1)
            
            # Average similarity (avoid division by zero)
            sims = torch.where(valid_counts > 0, cos_sums / valid_counts, 
                              torch.full_like(cos_sums, -1.0))
            
            # Update best similarities
            best_sims = torch.maximum(best_sims, sims)
        
        # Compute distances
        distances = 1.0 - best_sims
        distances = torch.where(best_sims < 0, torch.ones_like(distances), distances)
        
        # Transfer back to CPU
        return distances.cpu().numpy()
    
    def batch_distance_sc_gpu_optimized(query_sc, db_stack_gpu, db_norms_gpu):
        """
        Optimized GPU batch distance when database is already on GPU.
        
        Use this when processing multiple queries against the same database.
        
        Args:
            query_sc: Query ScanContext (ring_res, sector_res) - numpy array
            db_stack_gpu: Database SCs already on GPU (n_db, ring_res, sector_res) - torch tensor
            db_norms_gpu: Database norms already on GPU (n_db, sector_res) - torch tensor
            
        Returns:
            Array of distances (numpy)
        """
        device = db_stack_gpu.device
        query_gpu = torch.from_numpy(query_sc.astype(np.float32)).to(device)
        
        n_db = db_stack_gpu.shape[0]
        num_sectors = query_gpu.shape[1]
        
        best_sims = torch.full((n_db,), -1.0, dtype=torch.float32, device=device)
        
        for shift in range(num_sectors):
            query_shifted = torch.roll(query_gpu, shifts=shift, dims=1)
            query_norms_t = torch.linalg.norm(query_shifted, dim=0)
            
            query_valid = query_norms_t > 1e-8
            
            dot_products = torch.sum(query_shifted.unsqueeze(0) * db_stack_gpu, dim=1)
            denom = query_norms_t.unsqueeze(0) * db_norms_gpu
            valid_mask = query_valid.unsqueeze(0) & (db_norms_gpu > 1e-8)
            
            cos_sims = torch.where(valid_mask & (denom > 1e-8), 
                                   dot_products / (denom + 1e-8), 
                                   torch.zeros_like(dot_products))
            
            valid_counts = torch.sum(valid_mask.float(), dim=1)
            cos_sums = torch.sum(cos_sims, dim=1)
            
            sims = torch.where(valid_counts > 0, cos_sums / valid_counts, 
                              torch.full_like(cos_sums, -1.0))
            
            best_sims = torch.maximum(best_sims, sims)
        
        distances = 1.0 - best_sims
        distances = torch.where(best_sims < 0, torch.ones_like(distances), distances)
        
        return distances.cpu().numpy()
    
    def prepare_database_gpu(database_scs):
        """
        Prepare database for GPU processing.
        
        Call this once before processing multiple queries.
        
        Args:
            database_scs: List of database ScanContexts
            
        Returns:
            Tuple of (db_stack_gpu, db_norms_gpu) as PyTorch tensors on GPU
        """
        device = torch.device('cuda')
        db_stack = np.stack(database_scs, axis=0).astype(np.float32)
        db_stack_gpu = torch.from_numpy(db_stack).to(device)
        db_norms_gpu = torch.linalg.norm(db_stack_gpu, dim=1)
        return db_stack_gpu, db_norms_gpu
    
    print("[INFO] GPU batch functions available: batch_distance_sc_gpu, prepare_database_gpu")


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
                
                # Precompute column norms for faster distance computation
                sc_float = sc.astype(np.float64)
                col_norms = np.linalg.norm(sc_float, axis=0)
                
                # Store descriptor with precomputed norms
                self.descriptors[idx] = {
                    'd': sc.flatten().tolist(),  # Flatten for compatibility
                    'sc': sc,  # Keep 2D version for distance computation
                    'col_norms': col_norms  # Precomputed column norms
                }
                
            except Exception as e:
                self.logger.error(f"Failed to process sample {idx}: {e}")
                # Use zero descriptor as fallback
                self.descriptors[idx] = {
                    'd': np.zeros(self.sc_generator.ring_res * self.sc_generator.sector_res).tolist(),
                    'sc': np.zeros((self.sc_generator.ring_res, self.sc_generator.sector_res)),
                    'col_norms': np.zeros(self.sc_generator.sector_res)
                }
        
        return self.descriptors
    
    def compute_predictions(self, top_k=25, warmup_window=100, roi_window=50):
        """
        Compute top-k predictions for each query.
        
        Output format is compatible with PlaceRecognition.loop_closure_prediction
        for use with compute_recall_from_predictions.
        
        Args:
            top_k: Number of top candidates to retrieve
            warmup_window: Number of initial frames to skip
            roi_window: Window size to exclude nearby frames
            
        Returns:
            Dictionary with predictions in PlaceRecognition-compatible format:
            {
                'predictions': {query_idx: {
                    'candidates': [...], 'similarities': [...], 'positions_dist': [...],
                    'labels': [...], 'query_label': int, 'query_position': [...],
                    'gt_candidates': [...], 'gt_positions_dist': [...], 'gt_labels': [...]
                }},
                'query_indices': np.array([...]),
                'statistics': {...},
                'parameters': {...}
            }
        """
        n_samples = len(self.descriptors)
        
        # Get positions and labels from dataset
        positions = self.dataset.positions
        labels = self.dataset.labels
        
        # Make positions 2D (ignore z-coordinate for distance)
        positions_2d = positions.copy()
        if positions_2d.ndim == 2 and positions_2d.shape[1] >= 3:
            positions_2d[:, 2] = 0
        
        all_indices = np.arange(n_samples)
        predictions = {}
        query_indices = []
        
        self.logger.info(f"Computing predictions (top-{top_k}) with window={roi_window}...")
        self.logger.info(f"Total samples: {n_samples}, Warmup: {warmup_window}")
        
        # Use C++ accelerated version if available
        if USE_CPP:
            self.logger.info(f"Using C++ acceleration...")
            
            # Prepare data for C++ function
            all_scs = [self.descriptors[i]['sc'] for i in range(n_samples)]
            query_idx_list = list(range(warmup_window, n_samples))
            
            # Call C++ function
            cpp_predictions = scancontext_cpp.compute_top_k_predictions(
                all_scs, query_idx_list, roi_window, top_k
            )
            
            # Convert to PlaceRecognition-compatible format
            for query_idx, top_k_idx_list in cpp_predictions.items():
                query_position = positions_2d[query_idx]
                query_label = labels[query_idx]
                
                # Eligible indices (past frames outside ROI window)
                eligible_indices = all_indices[:query_idx - roi_window]
                eligible_positions = positions_2d[eligible_indices]
                eligible_labels = labels[eligible_indices]
                
                # Compute distances for the top-k predictions
                distances = [
                    distance_sc(self.descriptors[query_idx]['sc'], 
                               self.descriptors[idx]['sc'])
                    for idx in top_k_idx_list
                ]
                
                # Position distances for predictions
                pred_positions = positions_2d[top_k_idx_list]
                delta_pos = query_position - pred_positions
                position_distances = np.linalg.norm(delta_pos, axis=-1)
                
                pred_labels = labels[top_k_idx_list]
                
                # Ground truth: sort by position distance
                delta_pos_all = query_position - eligible_positions
                all_position_distances = np.linalg.norm(delta_pos_all, axis=-1)
                gt_sort_order = np.argsort(all_position_distances)
                
                gt_topk = min(top_k, len(eligible_indices))
                gt_topk_indices = gt_sort_order[:gt_topk]
                gt_candidates = eligible_indices[gt_topk_indices]
                gt_position_distances = all_position_distances[gt_topk_indices]
                gt_labels = eligible_labels[gt_topk_indices]
                
                predictions[query_idx] = {
                    'candidates': list(top_k_idx_list),
                    'similarities': distances,
                    'positions_dist': position_distances.tolist(),
                    'labels': pred_labels.tolist() if hasattr(pred_labels, 'tolist') else list(pred_labels),
                    'query_label': int(query_label),
                    'query_position': query_position.tolist(),
                    'gt_candidates': gt_candidates.tolist(),
                    'gt_positions_dist': gt_position_distances.tolist(),
                    'gt_labels': gt_labels.tolist()
                }
                query_indices.append(query_idx)
        elif USE_GPU:
            # GPU-accelerated implementation (PyTorch CUDA)
            self.logger.info(f"Using GPU acceleration (PyTorch CUDA)...")
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            
            # Prepare all ScanContexts - cache entire database on GPU
            all_scs = [self.descriptors[i]['sc'] for i in range(n_samples)]
            
            # Pre-transfer full database to GPU for maximum efficiency
            self.logger.info("Transferring database to GPU...")
            full_db_stack_gpu, full_db_norms_gpu = prepare_database_gpu(all_scs)
            
            for query_idx in tqdm(range(warmup_window, n_samples), desc="Computing predictions (GPU)"):
                query_sc = self.descriptors[query_idx]['sc']
                query_position = positions_2d[query_idx]
                query_label = labels[query_idx]
                
                # Eligible indices: past frames outside ROI window
                eligible_end = query_idx - roi_window
                
                if eligible_end <= 0:
                    continue
                
                eligible_indices = all_indices[:eligible_end]
                eligible_positions = positions_2d[eligible_indices]
                eligible_labels = labels[eligible_indices]
                
                # Use sliced GPU tensors for eligible entries (no CPU transfer!)
                db_stack_slice = full_db_stack_gpu[:eligible_end]
                db_norms_slice = full_db_norms_gpu[:eligible_end]
                
                # Compute distances using cached GPU data
                sc_distances = batch_distance_sc_gpu_optimized(query_sc, db_stack_slice, db_norms_slice)
                
                # Sort by descriptor distance (ascending)
                sort_order = np.argsort(sc_distances)
                
                # Get top-k predictions by descriptor similarity
                topk_actual = min(top_k, len(eligible_indices))
                topk_indices = sort_order[:topk_actual]
                
                predicted_candidates = eligible_indices[topk_indices]
                predicted_similarities = sc_distances[topk_indices]
                
                # Position distances for predictions
                pred_positions = eligible_positions[topk_indices]
                delta_pos = query_position - pred_positions
                position_distances = np.linalg.norm(delta_pos, axis=-1)
                
                predicted_labels = eligible_labels[topk_indices]
                
                # Ground truth: sort by position distance
                delta_pos_all = query_position - eligible_positions
                all_position_distances = np.linalg.norm(delta_pos_all, axis=-1)
                gt_sort_order = np.argsort(all_position_distances)
                
                gt_topk_indices = gt_sort_order[:topk_actual]
                gt_candidates = eligible_indices[gt_topk_indices]
                gt_position_distances = all_position_distances[gt_topk_indices]
                gt_labels = eligible_labels[gt_topk_indices]
                
                predictions[query_idx] = {
                    'candidates': predicted_candidates.tolist(),
                    'similarities': predicted_similarities.tolist(),
                    'positions_dist': position_distances.tolist(),
                    'labels': predicted_labels.tolist() if hasattr(predicted_labels, 'tolist') else list(predicted_labels),
                    'query_label': int(query_label),
                    'query_position': query_position.tolist(),
                    'gt_candidates': gt_candidates.tolist(),
                    'gt_positions_dist': gt_position_distances.tolist(),
                    'gt_labels': gt_labels.tolist()
                }
                query_indices.append(query_idx)
            
            # Free GPU memory
            del full_db_stack_gpu, full_db_norms_gpu
            torch.cuda.empty_cache()
        else:
            # Python fallback implementation (with batch optimization)
            self.logger.info(f"Using Python implementation...")
            
            # Check if batch_distance_sc is available and if we have precomputed norms
            use_batch = 'batch_distance_sc' in dir()
            has_norms = 'col_norms' in self.descriptors.get(0, {})
            
            if has_norms:
                self.logger.info("Using precomputed column norms for faster distance computation")
            
            for query_idx in tqdm(range(warmup_window, n_samples), desc="Computing predictions"):
                query_sc = self.descriptors[query_idx]['sc']
                query_position = positions_2d[query_idx]
                query_label = labels[query_idx]
                
                # Eligible indices: past frames outside ROI window
                eligible_indices = all_indices[:query_idx - roi_window]
                
                if len(eligible_indices) == 0:
                    continue
                
                eligible_positions = positions_2d[eligible_indices]
                eligible_labels = labels[eligible_indices]
                
                # Compute ScanContext distances to all eligible samples
                if use_batch and has_norms:
                    # Use batch processing with precomputed norms (fastest)
                    eligible_scs = [self.descriptors[idx]['sc'] for idx in eligible_indices]
                    query_norms = self.descriptors[query_idx]['col_norms']
                    db_norms = [self.descriptors[idx]['col_norms'] for idx in eligible_indices]
                    sc_distances = batch_distance_sc(query_sc, eligible_scs, query_norms, db_norms)
                elif use_batch:
                    # Use batch processing without precomputed norms
                    eligible_scs = [self.descriptors[idx]['sc'] for idx in eligible_indices]
                    sc_distances = batch_distance_sc(query_sc, eligible_scs)
                else:
                    # Fall back to individual distance computation
                    sc_distances = np.array([
                        distance_sc(query_sc, self.descriptors[idx]['sc']) 
                        for idx in eligible_indices
                    ])
                
                # Sort by descriptor distance (ascending)
                sort_order = np.argsort(sc_distances)
                
                # Get top-k predictions by descriptor similarity
                topk_actual = min(top_k, len(eligible_indices))
                topk_indices = sort_order[:topk_actual]
                
                predicted_candidates = eligible_indices[topk_indices]
                predicted_similarities = sc_distances[topk_indices]
                
                # Position distances for predictions
                pred_positions = eligible_positions[topk_indices]
                delta_pos = query_position - pred_positions
                position_distances = np.linalg.norm(delta_pos, axis=-1)
                
                predicted_labels = eligible_labels[topk_indices]
                
                # Ground truth: sort by position distance
                delta_pos_all = query_position - eligible_positions
                all_position_distances = np.linalg.norm(delta_pos_all, axis=-1)
                gt_sort_order = np.argsort(all_position_distances)
                
                gt_topk_indices = gt_sort_order[:topk_actual]
                gt_candidates = eligible_indices[gt_topk_indices]
                gt_position_distances = all_position_distances[gt_topk_indices]
                gt_labels = eligible_labels[gt_topk_indices]
                
                predictions[query_idx] = {
                    'candidates': predicted_candidates.tolist(),
                    'similarities': predicted_similarities.tolist(),
                    'positions_dist': position_distances.tolist(),
                    'labels': predicted_labels.tolist() if hasattr(predicted_labels, 'tolist') else list(predicted_labels),
                    'query_label': int(query_label),
                    'query_position': query_position.tolist(),
                    'gt_candidates': gt_candidates.tolist(),
                    'gt_positions_dist': gt_position_distances.tolist(),
                    'gt_labels': gt_labels.tolist()
                }
                query_indices.append(query_idx)
        
        # Compute statistics
        all_similarities = []
        all_position_dists = []
        for pred in predictions.values():
            all_similarities.extend(pred['similarities'])
            all_position_dists.extend(pred['positions_dist'])
        
        statistics = {
            'total_queries': len(query_indices),
            'topk': top_k,
            'window': roi_window,
            'similarity_metric': 'ScanContext',
            'avg_descriptor_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
            'avg_position_distance': float(np.mean(all_position_dists)) if all_position_dists else 0.0,
            'median_position_distance': float(np.median(all_position_dists)) if all_position_dists else 0.0
        }
        
        self.logger.info(f'Completed predictions for {len(query_indices)} queries')
        self.logger.info(f'Average descriptor distance: {statistics["avg_descriptor_similarity"]:.4f}')
        self.logger.info(f'Average position distance: {statistics["avg_position_distance"]:.2f}m')
        
        return {
            'predictions': predictions,
            'query_indices': np.array(query_indices),
            'statistics': statistics,
            'parameters': {
                'topk': top_k,
                'window': roi_window,
                'warmup': warmup_window,
                'sim_func': 'ScanContext'
            }
        }
    
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
                # Compute column norms for faster distance computation
                col_norms = np.linalg.norm(sc_2d.astype(np.float64), axis=0)
                
                self.descriptors[idx] = {
                    'd': flat_desc,
                    'sc': sc_2d,
                    'col_norms': col_norms
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
        Compute recall metrics using the shared PlaceRecognition interface.
        
        This method uses the same recall computation logic as PlaceRecognition,
        ensuring consistent evaluation across different methods.
        
        Args:
            predictions: Dictionary with predictions from compute_predictions
                         (must be in PlaceRecognition-compatible format)
            distance_thresholds: List of distance thresholds in meters
            top_k_list: List of top-k values to evaluate
            
        Returns:
            Dictionary with recall metrics
        """
        self.logger.info(f"Computing recall metrics using shared interface...")
        
        # Use local implementation that matches PlaceRecognition.compute_recall_from_predictions
        # This avoids torch dependency while using the same logic
        
        # Call the local computation method
        # The predictions dict is already in the correct format
        recall_results = self._compute_recall_from_predictions_impl(
            predictions, 
            distance_thresholds, 
            top_k_list
        )
        
        return recall_results
    
    def _compute_recall_from_predictions_impl(self, loop_closure_results: dict, 
                                               radius_thresholds: list,
                                               top_k_values: list = None) -> dict:
        """
        Compute recall from predictions - same logic as PlaceRecognition.compute_recall_from_predictions.
        
        This is a local implementation to avoid needing a full PlaceRecognition instance.
        """
        predictions = loop_closure_results['predictions']
        
        if top_k_values is None:
            top_k_values = [1, 5, 10, 25]

        top_k_range = list(range(1, max(top_k_values) + 1))

        # Ensure radius_thresholds is a list
        if not isinstance(radius_thresholds, list):
            radius_thresholds = [radius_thresholds]
            
        # Initialize result containers
        # Global metrics
        global_tp = {r: {k: 0 for k in top_k_range} for r in radius_thresholds}
        global_total = {r: {k: 0 for k in top_k_range} for r in radius_thresholds}
        
        # Segment-wise metrics
        segments = set()
        for pred in predictions.values():
            segments.add(pred['query_label'])
        
        segment_tp = {seg: {r: {k: 0 for k in top_k_range} for r in radius_thresholds} for seg in segments}
        segment_total = {seg: {r: {k: 0 for k in top_k_range} for r in radius_thresholds} for seg in segments}
        
        # Process each query prediction
        for query_idx, pred in predictions.items():
            query_label = pred['query_label']
            position_distances = np.array(pred['positions_dist'])
            candidate_labels = np.array(pred['labels'])
            
            # Ground truth info (sorted by position distance)
            gt_positions_dist = np.array(pred['gt_positions_dist'])
            gt_labels = np.array(pred['gt_labels'])
            
            # For each radius threshold
            for radius in radius_thresholds:
                # For each top-k value
                for k in top_k_range:
                    # ============================================================
                    # GROUND TRUTH CHECK (matches eval_row_place behavior)
                    # Check if a valid GT loop exists in top-k by POSITION
                    # A valid GT loop must be: within radius AND same segment label
                    # ============================================================
                    gt_topk_dists = gt_positions_dist[:k] if len(gt_positions_dist) >= k else gt_positions_dist
                    gt_topk_labels = gt_labels[:k] if len(gt_labels) >= k else gt_labels
                    
                    # Check if GT loop exists within radius for this segment
                    gt_in_range = gt_topk_dists <= radius
                    gt_same_segment = gt_topk_labels == query_label
                    gt_valid = gt_in_range & gt_same_segment
                    
                    if not np.any(gt_valid):
                        # No ground truth loop exists within this radius for this segment
                        # Skip this query - don't count it in recall calculation
                        continue
                    
                    # ============================================================
                    # PREDICTION CHECK
                    # Check if ANY of top-k PREDICTIONS is a true positive
                    # ============================================================
                    topk_dists = position_distances[:k] if len(position_distances) >= k else position_distances
                    topk_labels = candidate_labels[:k] if len(candidate_labels) >= k else candidate_labels
                    
                    # True positive: position distance <= radius AND same segment label
                    tp_mask = (topk_dists <= radius) & (topk_labels == query_label)
                    is_tp = np.any(tp_mask)
                    
                    # Update global counters
                    global_tp[radius][k] += int(is_tp)
                    global_total[radius][k] += 1
                    
                    # Update segment counters
                    segment_tp[query_label][radius][k] += int(is_tp)
                    segment_total[query_label][radius][k] += 1
        
        # Compute recall values
        global_recall = {}
        for radius in radius_thresholds:
            global_recall[radius] = {}
            for k in top_k_range:
                if global_total[radius][k] > 0:
                    global_recall[radius][k] = global_tp[radius][k] / global_total[radius][k]
                else:
                    global_recall[radius][k] = 0.0
        
        # Compute segment-wise recall
        segment_results = {}
        for seg in segments:
            segment_results[seg] = {
                'recall': {},
                'num_queries': sum(segment_total[seg][radius_thresholds[0]][k] for k in [top_k_range[0]])
            }
            for radius in radius_thresholds:
                segment_results[seg]['recall'][radius] = {}
                for k in top_k_range:
                    if segment_total[seg][radius][k] > 0:
                        segment_results[seg]['recall'][radius][k] = segment_tp[seg][radius][k] / segment_total[seg][radius][k]
                    else:
                        segment_results[seg]['recall'][radius][k] = 0.0
        
        # Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"RECALL RESULTS (with GT filtering)")
        self.logger.info(f"{'='*60}")
        
        for radius in radius_thresholds:
            self.logger.info(f"\nRadius {radius}m:")
            self.logger.info(f"  Valid queries: {global_total[radius][1]}")
            for k in top_k_values:
                recall = global_recall[radius][k]
                self.logger.info(f"  Recall@{k}: {recall:.4f}")
        
        return {
            'global': {
                'recall': global_recall,
                'num_queries': global_total[radius_thresholds[0]][1]
            },
            'segment': segment_results,
            'global_tp': global_tp,
            'global_total': global_total
        }
    
    
    
    
    def save_recall_csv(self, recall_results, save_dir, top_k_values=[1, 5, 10, 25]):
        """
        Save recall results to CSV files in PointNetGAP-compatible format.
        Saves both global recall and per-segment (row) recall.
        
        The main format is a matrix where:
        - Rows = top-k values (1, 2, ..., 25)
        - Columns = distance thresholds (meters)
        - Values = recall at that (top_k, distance) combination
        
        Args:
            recall_results: Dictionary with recall metrics from compute_recall.
                           Expected format: {'global': {'recall': {radius: {k: value}}}, 'segment': {...}}
            save_dir: Directory to save CSV files
            top_k_values: List of top-k values to save (default: [1, 5, 10, 25])
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        global_recall = recall_results['global']['recall']
        
        # Extract distance thresholds and top-k list from the recall dict
        distance_thresholds = sorted(global_recall.keys())
        
        # Get all top-k values from the recall dict
        if distance_thresholds:
            all_topk = sorted(global_recall[distance_thresholds[0]].keys())
        else:
            self.logger.warning("No distance thresholds found in recall results")
            return
        
        # ============================================================
        # Save GLOBAL recall matrix
        # ============================================================
        recall_matrix = []
        for k in all_topk:
            row = []
            for dist in distance_thresholds:
                if k in global_recall[dist]:
                    recall = global_recall[dist][k]
                    row.append(recall)
                else:
                    row.append(0.0)
            recall_matrix.append(row)
        
        # Create DataFrame with distance thresholds as column names
        df_recall = pd.DataFrame(recall_matrix, columns=distance_thresholds)
        df_recall.index = all_topk
        
        # Save main recall.csv file (PointNetGAP format)
        csv_path = save_dir / 'recall.csv'
        df_recall.to_csv(csv_path)
        self.logger.info(f"Saved global recall matrix to: {csv_path}")
        
        # ============================================================
        # Save summary at specific top-k values
        # ============================================================
        summary_path = save_dir / 'recall_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("RECALL SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            for dist in distance_thresholds:
                f.write(f"Distance Threshold: {dist}m\n")
                f.write("-"*40 + "\n")
                for k in top_k_values:
                    if k in global_recall[dist]:
                        recall = global_recall[dist][k]
                        f.write(f"  Recall@{k}: {recall:.4f}\n")
                f.write("\n")
        
        self.logger.info(f"Saved recall summary to: {summary_path}")

        
        # ============================================================
        # Save PER-SEGMENT recall matrices
        # ============================================================
        if 'segment' in recall_results:
            segment_results = recall_results['segment']
            unique_labels = sorted(segment_results.keys())
            self.logger.info(f"Saving per-segment recall for {len(unique_labels)} segments: {unique_labels}")
            
            for label in unique_labels:
                seg_recall = segment_results[label]['recall']
                
                # Create recall matrix for this segment
                segment_matrix = []
                for k in all_topk:
                    row = []
                    for dist in distance_thresholds:
                        if dist in seg_recall and k in seg_recall[dist]:
                            recall = seg_recall[dist][k]
                            row.append(recall)
                        else:
                            row.append(0.0)
                    segment_matrix.append(row)
                
                # Create DataFrame
                df_segment = pd.DataFrame(segment_matrix, columns=distance_thresholds)
                df_segment.index = all_topk
                
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
        pr_system.save_recall_csv(recall_results, output_dir, top_k_values=top_k_eval)
        
        # Log recall results - new format uses numeric keys
        logger.info("="*80)
        logger.info("Recall Results:")
        logger.info("="*80)
        global_recall = recall_results['global']['recall']
        for dist in sorted(global_recall.keys()):
            # Only log select distances to avoid spam
            if dist in [5, 10, 15, 20]:
                logger.info(f"\nDistance threshold: {dist}m")
                for k in [1, 5, 10, 25]:
                    if k in global_recall[dist]:
                        recall_val = global_recall[dist][k]
                        logger.info(f"  Recall@{k}: {recall_val:.4f}")
        
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
