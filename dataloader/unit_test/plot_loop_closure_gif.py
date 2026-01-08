"""
Generate animated GIF showing loop closure detection frame by frame.

This script creates a visualization that shows:
1. The trajectory path
2. Current query frame (red)
3. Loop closure matches found (green connections)
4. Historical trajectory colored by label
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path


class LoopClosureVisualizer:
    """Visualize loop closure detection frame by frame."""
    
    def __init__(self, fs: file_structure, gt_loop_closures: dict, seq_name: str, 
                 sample_rate: int = 5):
        """
        Initialize the visualizer.
        
        Args:
            fs: file_structure object with dataset
            gt_loop_closures: Ground truth loop closure dict from get_ground_truth_loop_closure()
            seq_name: Sequence name for title
            sample_rate: Show every Nth frame across entire path (default: 5)
        """
        self.fs = fs
        self.gt_lc = gt_loop_closures
        self.seq_name = seq_name
        self.sample_rate = sample_rate
        
        # Get data
        self.positions = fs._get_positions_()
        self.labels = fs._get_labels()
        self.frame_ids = fs._get_frame_ids()
        
        # Align and elevate positions
        self.aligned_positions = aligned_path(self.positions)
        self.elevated_positions = elevate_along_path(self.aligned_positions, max_elevation=20.0)
        
        # Sample frames uniformly across entire path
        self.frame_indices = np.arange(0, len(self.positions), sample_rate)
        
        # Color mapping
        self.colors = generate_label_colors(max(self.labels) + 1)
        self.label_colors = [self.colors[int(label)] for label in self.labels]
        
        # Build query to neighbors map from ground truth
        self.query_neighbors_map = {}
        for q_idx, n_idx, dist in zip(
            self.gt_lc['query_indices'],
            self.gt_lc['neighbor_indices'],
            self.gt_lc['distances']
        ):
            if q_idx not in self.query_neighbors_map:
                self.query_neighbors_map[q_idx] = []
            self.query_neighbors_map[q_idx].append({
                'neighbor_idx': int(n_idx),
                'distance': float(dist)
            })
    
    def get_frame_data(self, query_idx: int):
        """Get visualization data for a specific query frame."""
        neighbors = self.query_neighbors_map.get(query_idx, [])
        neighbor_indices = [n['neighbor_idx'] for n in neighbors]
        neighbor_distances = [n['distance'] for n in neighbors]
        
        return {
            'query_idx': int(query_idx),
            'neighbor_indices': neighbor_indices,
            'neighbor_distances': neighbor_distances,
            'num_neighbors': len(neighbors)
        }
    
    def create_figure(self, figsize=(16, 12)):
        """Create figure with 3D axis."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        return fig, ax
    
    def plot_trajectory(self, ax, alpha=0.3):
        """Plot the entire trajectory."""
        ax.plot(self.elevated_positions[:, 0], 
               self.elevated_positions[:, 1], 
               self.elevated_positions[:, 2],
               'k-', alpha=alpha, linewidth=1.0, label='Path trajectory')
    
    def plot_positions(self, ax, alpha=0.2, s=10):
        """Plot all positions colored by label."""
        scatter = ax.scatter(self.elevated_positions[:, 0], 
                            self.elevated_positions[:, 1], 
                            self.elevated_positions[:, 2],
                            c=self.label_colors, s=s, alpha=alpha, edgecolors='none')
        return scatter
    
    def plot_frame(self, ax, query_idx: int, frame_data: dict, 
                   show_connections: bool = True, show_trajectory: bool = True):
        """
        Plot a single frame with loop closures.
        
        Args:
            ax: Matplotlib 3D axis
            query_idx: Index of query frame
            frame_data: Data from get_frame_data()
            show_connections: Whether to show lines to neighbors
            show_trajectory: Whether to show trajectory up to this point
        """
        # Clear previous plot elements (but keep background)
        ax.clear()
        
        # Plot trajectory
        if show_trajectory:
            self.plot_trajectory(ax, alpha=0.2)
        
        # Plot all positions faintly
        self.plot_positions(ax, alpha=0.15, s=5)
        
        # Highlight trajectory up to current frame
        current_frame_mask = self.frame_ids <= self.frame_ids[query_idx]
        current_trajectory = self.elevated_positions[current_frame_mask]
        ax.plot(current_trajectory[:, 0], 
               current_trajectory[:, 1], 
               current_trajectory[:, 2],
               'b-', alpha=0.6, linewidth=2.0, label='Trajectory up to query')
        
        # Plot query position (red)
        query_pos = self.elevated_positions[query_idx]
        ax.scatter(*query_pos, color='red', s=300, alpha=1.0, marker='*', 
                  edgecolors='darkred', linewidth=2, label='Query frame', zorder=100)
        
        # Plot neighbors (green) and connections
        if show_connections and frame_data['neighbor_indices']:
            neighbor_indices = frame_data['neighbor_indices']
            neighbor_distances = frame_data['neighbor_distances']
            
            for neighbor_idx, distance in zip(neighbor_indices, neighbor_distances):
                nn_pos = self.elevated_positions[neighbor_idx]
                
                # Plot neighbor
                ax.scatter(*nn_pos, color='lime', s=200, alpha=0.8, marker='o',
                          edgecolors='darkgreen', linewidth=1.5, zorder=50)
                
                # Draw connection line
                ax.plot([query_pos[0], nn_pos[0]],
                       [query_pos[1], nn_pos[1]],
                       [query_pos[2], nn_pos[2]],
                       'g-', alpha=0.6, linewidth=2.0, zorder=30)
        
        # Set labels and title
        ax.set_xlabel("X (m)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (m)", fontsize=11, fontweight='bold')
        ax.set_zlabel("Z (Elevation, m)", fontsize=11, fontweight='bold')
        
        title = f"{self.seq_name} - Loop Closure Detection\n"
        title += f"Frame {query_idx} | "
        title += f"Loop closures found: {frame_data['num_neighbors']}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Set equal aspect ratio
        max_range = np.array([self.elevated_positions[:, 0].max() - self.elevated_positions[:, 0].min(),
                             self.elevated_positions[:, 1].max() - self.elevated_positions[:, 1].min(),
                             self.elevated_positions[:, 2].max() - self.elevated_positions[:, 2].min()]).max() / 2.0
        
        mid_x = (self.elevated_positions[:, 0].max() + self.elevated_positions[:, 0].min()) * 0.5
        mid_y = (self.elevated_positions[:, 1].max() + self.elevated_positions[:, 1].min()) * 0.5
        mid_z = (self.elevated_positions[:, 2].max() + self.elevated_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add legend
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        
        # Add statistics text
        stats_text = (f"Total frames: {len(self.positions)}\n"
                     f"Frame ID: {self.frame_ids[query_idx]}\n"
                     f"Label: {int(self.labels[query_idx])}\n"
                     f"Neighbors found: {frame_data['num_neighbors']}")
        ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add distance info if neighbors exist
        if frame_data['neighbor_indices']:
            dist_text = "Distances (m):\n"
            for i, dist in enumerate(frame_data['neighbor_distances'][:5]):  # Show top 5
                dist_text += f"  {i+1}. {dist:.4f}\n"
            if len(frame_data['neighbor_distances']) > 5:
                dist_text += f"  ... +{len(frame_data['neighbor_distances']) - 5} more"
            ax.text2D(0.98, 0.95, dist_text, transform=ax.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    def generate_gif(self, output_path: str = None, fps: int = 2, dpi: int = 100):
        """
        Generate animated GIF showing loop closure detection.
        
        Args:
            output_path: Path to save GIF (default: ./loop_closure_{seq_name}.gif)
            fps: Frames per second in GIF
            dpi: DPI for image quality
            
        Returns:
            Path to saved GIF
        """
        if output_path is None:
            output_path = f"loop_closure_{self.seq_name}.gif"
        
        # Ensure output directory exists
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating GIF: {output_path}")
        print(f"Total frames to visualize: {len(self.frame_indices)}")
        
        # Create figure
        fig, ax = self.create_figure(figsize=(14, 10))
        
        # Generate frames
        frame_images = []
        for frame_num, frame_idx in enumerate(self.frame_indices):
            print(f"  Rendering frame {frame_num + 1}/{len(self.frame_indices)} (frame index: {frame_idx})", end='\r')
            
            frame_data = self.get_frame_data(frame_idx)
            self.plot_frame(ax, frame_idx, frame_data, show_connections=True, show_trajectory=True)
            
            # Convert figure to image array
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame_images.append(image)
        
        print(f"\n  Rendering complete! Saving to disk...")
        
        # Save as GIF
        # Convert fps to duration in milliseconds (duration = 1000 / fps)
        import imageio
        duration_ms = int(1000 / fps) if fps > 0 else 500  # Default 500ms if fps is 0
        imageio.mimsave(output_path, frame_images, duration=duration_ms)
        
        plt.close(fig)
        
        print(f"✓ GIF saved: {output_path}")
        print(f"  - Size: {len(frame_images)} frames")
        print(f"  - Frame rate: {fps} fps (duration: {duration_ms}ms per frame)")
        print(f"  - Total duration: {len(frame_images) * duration_ms / 1000:.1f} seconds")
        
        return output_path


def generate_loop_closure_gifs(root_dir: str, sequences: list, output_dir: str = None,
                               warm_up: int = 100, lower_bound_idx: int = 50,
                               distance_threshold: float = 10.0, topk: int = None,
                               sample_rate: int = 10):
    """
    Generate loop closure GIFs for multiple sequences.
    
    Args:
        root_dir: Root dataset directory
        sequences: List of sequence names
        output_dir: Directory to save GIFs (default: root_dir)
        warm_up: Warmup frames to skip
        lower_bound_idx: Minimum frame gap
        distance_threshold: Distance threshold for loop closure
        topk: Top-k neighbors to keep
        sample_rate: Skip every N frames when sampling for visualization
    """
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(root_dir, 'loop_closure_gifs')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"GIFs will be saved to: {output_dir}")
    
    for seq in sequences:
        print(f"\n{'='*80}")
        print(f"Processing sequence: {seq}")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            fs = file_structure(root_dir, seq, verbose=False)
            
            print(f"Dataset loaded:")
            print(f"  Total positions: {len(fs._get_positions_())}")
            print(f"  Unique labels: {np.unique(fs._get_labels())}")
            
            # Compute ground truth loop closures
            print(f"\nComputing ground truth loop closures...")
            gt_lc = fs.get_ground_truth_loop_closure(
                warm_up=warm_up,
                lower_bound_idx=lower_bound_idx,
                distance_threshold=distance_threshold,
                topk=topk
            )
            
            print(f"  Total loop closures: {gt_lc['valid_loop_closures']}")
            print(f"  Unique query frames: {len(np.unique(gt_lc['query_indices']))}")
            
            # Create visualizer and generate GIF
            visualizer = LoopClosureVisualizer(
                fs, gt_lc, seq,
                sample_rate=sample_rate
            )
            
            output_path = os.path.join(output_dir, f"loop_closure_{seq}.gif")
            visualizer.generate_gif(output_path, fps=2, dpi=80)
            
            print(f"✓ Successfully generated GIF for {seq}")
            print(f"  Saved to: {output_path}\n")
            
        except Exception as e:
            import traceback
            print(f"✗ Error processing {seq}: {e}")
            traceback.print_exc()
            print()
            continue


if __name__ == '__main__':
    # Configuration
    ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"
    OUTPUT_DIR = ROOT_DIR  # Where to save GIFs
    SEQUENCES = ["PCD_EASY", "PCD_Easy_DARK", "PCD_MED", "PCD_RAS_EASY"]
    
    # Parameters for ground truth computation
    WARM_UP = 100
    LOWER_BOUND_IDX = 50
    DISTANCE_THRESHOLD = 10.0
    TOPK = 1
    
    # Visualization parameter - skip every N frames
    SAMPLE_RATE = 10  # Show every 10th frame in the GIF
    
    print("Loop Closure Detection GIF Generator")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Root directory: {ROOT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sequences: {SEQUENCES}")
    print(f"  Warm-up: {WARM_UP}")
    print(f"  Lower bound index: {LOWER_BOUND_IDX}")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD}")
    print(f"  Top-K: {TOPK}")
    print(f"  Sample rate: {SAMPLE_RATE} (show every {SAMPLE_RATE}th frame)")
    print("=" * 80)
    
    # Generate GIFs
    generate_loop_closure_gifs(
        ROOT_DIR,
        SEQUENCES,
        output_dir=OUTPUT_DIR,
        warm_up=WARM_UP,
        lower_bound_idx=LOWER_BOUND_IDX,
        distance_threshold=DISTANCE_THRESHOLD,
        topk=TOPK,
        sample_rate=SAMPLE_RATE
    )
    
    print("\n" + "=" * 80)
    print("All GIFs generated successfully!")
    print("=" * 80)
