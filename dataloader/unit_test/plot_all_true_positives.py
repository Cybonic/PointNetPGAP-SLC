"""
Plot all true positive loop closures on the 3D trajectory path.

This script creates a static visualization showing:
1. The complete trajectory path
2. All true positive loop closure connections
3. Query points and their matched neighbors
4. Color-coded segments

Similar to the ground truth visualization but using model predictions.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path


class TruePositivesVisualizer:
    """Visualize all true positive loop closures on the trajectory."""
    
    def __init__(self, fs: file_structure, predictions: dict, seq_name: str,
                 topk: int = 1, distance_threshold: float = 10.0,
                 min_temporal_distance: int = 50):
        """
        Initialize the visualizer.
        
        Args:
            fs: file_structure object with dataset
            predictions: Dictionary of predictions loaded from predictions.pkl
            seq_name: Sequence name for title
            topk: Top-K predictions to consider (default: 1)
            distance_threshold: Maximum distance for valid loop closures (meters)
            min_temporal_distance: Minimum frame distance to consider as loop closure
        """
        self.fs = fs
        self.predictions = predictions
        self.seq_name = seq_name
        self.topk = topk
        self.distance_threshold = distance_threshold
        self.min_temporal_distance = min_temporal_distance
        
        # Get data
        self.positions = fs._get_positions_()
        self.labels = fs._get_labels()
        self.frame_ids = fs._get_frame_ids()
        
        # Align and elevate positions
        self.aligned_positions = aligned_path(self.positions)
        self.elevated_positions = elevate_along_path(self.aligned_positions, max_elevation=20.0)
        
        # Color mapping
        self.colors = generate_label_colors(max(self.labels) + 1)
        self.label_colors = [self.colors[int(label)] for label in self.labels]
        
    def collect_all_true_positives(self):
        """
        Collect all true positive loop closures from predictions.
        
        Returns:
            List of tuples: (query_idx, neighbor_idx, distance, query_segment, neighbor_segment)
        """
        true_positives = []
        
        for query_idx in sorted(self.predictions.keys()):
            pred_data = self.predictions[query_idx]
            pred_loops = pred_data['pred_loops']
            query_segment = pred_data['segment']
            
            # Get top-k predictions
            pred_indices = pred_loops['idx'][:self.topk]
            pred_distances = pred_loops['dist'][:self.topk]
            pred_segments = pred_loops['segment'][:self.topk]
            
            # Filter by distance threshold
            valid_mask = pred_distances <= self.distance_threshold
            valid_indices = pred_indices[valid_mask]
            valid_distances = pred_distances[valid_mask]
            valid_segments = pred_segments[valid_mask]
            
            # Filter by temporal distance
            temporal_distances = np.abs(query_idx - valid_indices)
            temporal_mask = temporal_distances >= self.min_temporal_distance
            valid_indices = valid_indices[temporal_mask]
            valid_distances = valid_distances[temporal_mask]
            valid_segments = valid_segments[temporal_mask]
            
            # Identify true positives (same segment)
            is_tp = (valid_segments == query_segment)
            
            for neighbor_idx, distance, neighbor_segment, is_positive in zip(
                valid_indices, valid_distances, valid_segments, is_tp
            ):
                if is_positive:
                    true_positives.append((
                        int(query_idx),
                        int(neighbor_idx),
                        float(distance),
                        int(query_segment),
                        int(neighbor_segment)
                    ))
        
        return true_positives
    
    def plot_all_true_positives(self, save_path: str = None, figsize=(20, 16),
                                view_angle=(30, 45), show_legend=True,
                                connection_alpha=0.8, connection_linewidth=2.0):
        """
        Plot all true positive loop closures on the trajectory.
        
        Args:
            save_path: Path to save the figure (if None, will display)
            figsize: Figure size (width, height)
            view_angle: 3D view angle (elevation, azimuth)
            show_legend: Whether to show legend
            connection_alpha: Alpha value for connection lines
            connection_linewidth: Line width for connections
        """
        # Collect all true positives
        true_positives = self.collect_all_true_positives()
        
        print(f"\nFound {len(true_positives)} true positive loop closures")
        print(f"Parameters: top-{self.topk}, distance<={self.distance_threshold}m, "
              f"temporal_dist>={self.min_temporal_distance}")
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory line in black (thicker and more prominent)
        ax.plot(self.elevated_positions[:, 0],
               self.elevated_positions[:, 1],
               self.elevated_positions[:, 2],
               'k-', alpha=1.0, linewidth=2.5, label='Trajectory Path', zorder=1)
        
        # Plot all true positive connections in GREEN
        query_positions = []
        neighbor_positions = []
        
        for query_idx, neighbor_idx, distance, _, _ in true_positives:
            query_pos = self.elevated_positions[query_idx]
            neighbor_pos = self.elevated_positions[neighbor_idx]
            
            query_positions.append(query_pos)
            neighbor_positions.append(neighbor_pos)
            
            # Draw connection line in GREEN
            ax.plot([query_pos[0], neighbor_pos[0]],
                   [query_pos[1], neighbor_pos[1]],
                   [query_pos[2], neighbor_pos[2]],
                   'g-', alpha=connection_alpha, linewidth=connection_linewidth, zorder=10,
                   label='Loop Closures (TP)' if len(query_positions) == 1 else '')
        
        # Highlight query and neighbor points (optional, can be removed for cleaner look)
        if len(query_positions) > 0:
            query_positions = np.array(query_positions)
            neighbor_positions = np.array(neighbor_positions)
            
            # Plot unique query points in red
            unique_queries = np.unique(query_positions, axis=0)
            ax.scatter(unique_queries[:, 0], unique_queries[:, 1], unique_queries[:, 2],
                      color='red', s=30, alpha=0.7, marker='o',
                      edgecolors='darkred', linewidth=0.5, zorder=50)
            
            # Plot unique neighbor points in green
            unique_neighbors = np.unique(neighbor_positions, axis=0)
            ax.scatter(unique_neighbors[:, 0], unique_neighbors[:, 1], unique_neighbors[:, 2],
                      color='lime', s=30, alpha=0.7, marker='o',
                      edgecolors='darkgreen', linewidth=0.5, zorder=50)
        
        # Set labels
        ax.set_xlabel("X (m)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Y (m)", fontsize=14, fontweight='bold')
        ax.set_zlabel("Z (Elevation, m)", fontsize=14, fontweight='bold')
        
        # Set title
        title = f"{self.seq_name} - True Positive Loop Closures\n"
        title += f"Total TP: {len(true_positives)} | Top-{self.topk} | Distance ≤ {self.distance_threshold}m"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Set view angle
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Set equal aspect ratio
        max_range = np.array([
            self.elevated_positions[:, 0].max() - self.elevated_positions[:, 0].min(),
            self.elevated_positions[:, 1].max() - self.elevated_positions[:, 1].min(),
            self.elevated_positions[:, 2].max() - self.elevated_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (self.elevated_positions[:, 0].max() + self.elevated_positions[:, 0].min()) * 0.5
        mid_y = (self.elevated_positions[:, 1].max() + self.elevated_positions[:, 1].min()) * 0.5
        mid_z = (self.elevated_positions[:, 2].max() + self.elevated_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Show legend
        if show_legend:
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
            print(f"\nFigure saved to: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
        
        return true_positives


def main():
    """Main function to generate true positives visualization."""
    
    # Configuration
    dataset_root = "/home/tiago/workspace/place_uk/dataset/place_v2/PlaceRecognitionTestPolyTunnel"
    predictions_path = "/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2/PCD_MED/PointNetPGAP-None/predictions/place/0.533@1/predictions.pkl"
    output_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives"
    
    sequence = "PCD_MED"
    model_name = "PointNetPGAP"
    
    # Parameters
    topk = 1
    distance_threshold = 10.0  # meters
    min_temporal_distance = 50  # frames
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"True Positive Loop Closures Visualization")
    print("=" * 80)
    print(f"Dataset: {dataset_root}")
    print(f"Sequence: {sequence}")
    print(f"Model: {model_name}")
    print(f"Predictions: {predictions_path}")
    print(f"Top-K: {topk}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Min temporal distance: {min_temporal_distance} frames")
    print("=" * 80)
    
    # Load predictions
    print("\nLoading predictions...")
    with open(predictions_path, 'rb') as f:
        predictions = pickle.load(f)
    print(f"Loaded {len(predictions)} predictions")
    
    # Load dataset
    print("\nLoading dataset...")
    fs = file_structure(dataset_root, sequence)
    print(f"Dataset loaded: {len(fs._get_positions_())} frames")
    
    # Create visualizer
    visualizer = TruePositivesVisualizer(
        fs=fs,
        predictions=predictions,
        seq_name=f"{sequence} - {model_name}",
        topk=topk,
        distance_threshold=distance_threshold,
        min_temporal_distance=min_temporal_distance
    )
    
    # Generate visualization
    output_path = os.path.join(output_dir, f"{sequence}_{model_name}_all_tp.png")
    
    print("\nGenerating visualization...")
    true_positives = visualizer.plot_all_true_positives(
        save_path=output_path,
        figsize=(20, 16),
        view_angle=(30, 45),
        show_legend=True,
        connection_alpha=0.8,  # More visible green lines
        connection_linewidth=2.0  # Thicker green lines
    )
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Statistics")
    print("=" * 80)
    print(f"Total true positive loop closures: {len(true_positives)}")
    
    if len(true_positives) > 0:
        distances = [tp[2] for tp in true_positives]
        print(f"Distance statistics:")
        print(f"  Min: {np.min(distances):.2f}m")
        print(f"  Max: {np.max(distances):.2f}m")
        print(f"  Mean: {np.mean(distances):.2f}m")
        print(f"  Median: {np.median(distances):.2f}m")
        
        # Count unique queries
        unique_queries = len(set([tp[0] for tp in true_positives]))
        print(f"\nUnique query frames with TP: {unique_queries}")
        
        # Count by segment
        segments = {}
        for query_idx, _, _, query_seg, _ in true_positives:
            segments[query_seg] = segments.get(query_seg, 0) + 1
        print(f"\nTrue positives by segment:")
        for seg in sorted(segments.keys()):
            print(f"  Segment {seg}: {segments[seg]} TPs")
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
