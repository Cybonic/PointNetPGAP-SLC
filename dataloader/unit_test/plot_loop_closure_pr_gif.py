"""
Generate animated GIF showing loop closure detection frame by frame using PREDICTIONS.

This script creates a visualization that shows:
1. The trajectory path
2. Current query frame (red)
3. Loop closure matches found from predictions (green connections)
4. Historical trajectory colored by label

This version uses actual model predictions instead of ground truth.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path


class LoopClosureVisualizerPredictions:
    """Visualize loop closure detection frame by frame using model predictions."""
    
    def __init__(self, fs: file_structure, predictions: dict, seq_name: str, 
                 topk: int = 1, distance_threshold: float = 10.0,
                 sample_rate: int = 5):
        """
        Initialize the visualizer with predictions.
        
        Args:
            fs: file_structure object with dataset
            predictions: Dictionary of predictions loaded from predictions.pkl
                         Format: {query_idx: {'pred_loops': {'idx': [], 'dist': [], 'segment': []}, ...}}
            seq_name: Sequence name for title
            topk: Top-K predictions to visualize (default: 1)
            distance_threshold: Maximum distance for valid loop closures (meters)
            sample_rate: Show every Nth frame across entire path (default: 5)
        """
        self.fs = fs
        self.predictions = predictions
        self.seq_name = seq_name
        self.topk = topk
        self.distance_threshold = distance_threshold
        self.sample_rate = sample_rate
        
        # Get data
        self.positions = fs._get_positions_()
        self.labels = fs._get_labels()
        self.frame_ids = fs._get_frame_ids()
        
        # Align and elevate positions
        self.aligned_positions = aligned_path(self.positions)
        self.elevated_positions = elevate_along_path(self.aligned_positions, max_elevation=20.0)
        
        # Get available query indices from predictions
        available_queries = sorted(list(predictions.keys()))
        
        # Sample frames uniformly from available queries only
        if len(available_queries) > 0:
            # Sample every N-th query from available predictions
            sample_step = max(1, sample_rate)
            self.frame_indices = available_queries[::sample_step]
        else:
            self.frame_indices = []
            print("WARNING: No predictions available!")
        
        # Color mapping
        self.colors = generate_label_colors(max(self.labels) + 1)
        self.label_colors = [self.colors[int(label)] for label in self.labels]
    
    def get_frame_data(self, query_idx: int):
        """
        Get visualization data for a specific query frame from predictions.
        
        Args:
            query_idx: Query frame index
            
        Returns:
            Dictionary with neighbor information from predictions including TP/FP classification
        """
        if query_idx not in self.predictions:
            return {
                'query_idx': int(query_idx),
                'neighbor_indices': [],
                'neighbor_distances': [],
                'neighbor_segments': [],
                'is_true_positive': [],
                'num_neighbors': 0,
                'num_true_positives': 0,
                'num_false_positives': 0
            }
        
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
        
        # Determine if each prediction is a true positive or false positive
        # TP: predicted segment matches query segment
        # FP: predicted segment does not match query segment
        is_true_positive = (valid_segments == query_segment)
        
        return {
            'query_idx': int(query_idx),
            'query_segment': int(query_segment),
            'neighbor_indices': valid_indices.tolist(),
            'neighbor_distances': valid_distances.tolist(),
            'neighbor_segments': valid_segments.tolist(),
            'is_true_positive': is_true_positive.tolist(),
            'num_neighbors': len(valid_indices),
            'num_true_positives': int(np.sum(is_true_positive)),
            'num_false_positives': int(np.sum(~is_true_positive))
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
        Plot a single frame with loop closures from predictions.
        
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
        
        # Calculate max_range for label offset (needed before plotting neighbors)
        max_range = np.array([self.elevated_positions[:, 0].max() - self.elevated_positions[:, 0].min(),
                             self.elevated_positions[:, 1].max() - self.elevated_positions[:, 1].min(),
                             self.elevated_positions[:, 2].max() - self.elevated_positions[:, 2].min()]).max() / 2.0
        
        # Plot neighbors (green for TP, red for FP) and connections from predictions
        if show_connections and frame_data['neighbor_indices']:
            neighbor_indices = frame_data['neighbor_indices']
            neighbor_distances = frame_data['neighbor_distances']
            is_true_positive = frame_data['is_true_positive']
            
            for neighbor_idx, distance, is_tp in zip(neighbor_indices, neighbor_distances, is_true_positive):
                nn_pos = self.elevated_positions[neighbor_idx]
                
                # Color: green for true positive, red for false positive
                if is_tp:
                    color = 'lime'
                    edge_color = 'darkgreen'
                    line_color = 'g'
                    label_text = 'TP'
                    label_color = 'white'
                    label_bg = 'green'
                else:
                    color = 'red'
                    edge_color = 'darkred'
                    line_color = 'r'
                    label_text = 'FP'
                    label_color = 'white'
                    label_bg = 'red'
                
                # Plot neighbor
                ax.scatter(*nn_pos, color=color, s=200, alpha=0.8, marker='o',
                          edgecolors=edge_color, linewidth=1.5, zorder=50)
                
                # Draw connection line
                ax.plot([query_pos[0], nn_pos[0]],
                       [query_pos[1], nn_pos[1]],
                       [query_pos[2], nn_pos[2]],
                       line_color, alpha=0.6, linewidth=2.0, zorder=30)
                
                # Add TP/FP label near the neighbor point
                # Offset the label slightly above the point
                label_offset = max_range * 0.02  # Small offset for visibility
                ax.text(nn_pos[0], nn_pos[1], nn_pos[2] + label_offset,
                       label_text, fontsize=9, fontweight='bold',
                       color=label_color, ha='center', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=label_bg, alpha=0.8, edgecolor='white'),
                       zorder=60)
        
        # Set labels and title
        ax.set_xlabel("X (m)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Y (m)", fontsize=11, fontweight='bold')
        ax.set_zlabel("Z (Elevation, m)", fontsize=11, fontweight='bold')
        
        title = f"{self.seq_name} - Loop Closure Detection (PREDICTIONS)\n"
        title += f"Frame {query_idx} | Top-{self.topk} | "
        title += f"TP: {frame_data['num_true_positives']} | FP: {frame_data['num_false_positives']}"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Set equal aspect ratio (max_range already calculated above)
        mid_x = (self.elevated_positions[:, 0].max() + self.elevated_positions[:, 0].min()) * 0.5
        mid_y = (self.elevated_positions[:, 1].max() + self.elevated_positions[:, 1].min()) * 0.5
        mid_z = (self.elevated_positions[:, 2].max() + self.elevated_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add custom legend for TP/FP
        from matplotlib.lines import Line2D
        custom_lines = [
            Line2D([0], [0], color='b', linewidth=2, alpha=0.6, label='Trajectory up to query'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, markeredgecolor='darkred', markeredgewidth=2, label='Query frame'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
                   markersize=10, markeredgecolor='darkgreen', markeredgewidth=1.5, label='True Positive (TP)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, markeredgecolor='darkred', markeredgewidth=1.5, label='False Positive (FP)')
        ]
        ax.legend(handles=custom_lines, loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.2)
        
        # Add statistics text
        stats_text = (f"Total frames: {len(self.positions)}\n"
                     f"Frame ID: {self.frame_ids[query_idx]}\n"
                     f"Query Segment: {frame_data['query_segment']}\n"
                     f"Top-K: {self.topk}\n"
                     f"Dist threshold: {self.distance_threshold}m\n"
                     f"True Positives: {frame_data['num_true_positives']}\n"
                     f"False Positives: {frame_data['num_false_positives']}")
        ax.text2D(0.02, 0.95, stats_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add distance and segment info if neighbors exist
        if frame_data['neighbor_indices']:
            dist_text = "Predictions:\n"
            for i, (dist, seg, is_tp) in enumerate(zip(
                frame_data['neighbor_distances'][:5],
                frame_data['neighbor_segments'][:5],
                frame_data['is_true_positive'][:5]
            )):
                tp_label = "TP" if is_tp else "FP"
                dist_text += f"  {i+1}. {dist:.4f}m | Seg:{seg} | {tp_label}\n"
            if len(frame_data['neighbor_distances']) > 5:
                dist_text += f"  ... +{len(frame_data['neighbor_distances']) - 5} more"
            ax.text2D(0.98, 0.95, dist_text, transform=ax.transAxes,
                     fontsize=9, verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    def generate_gif(self, output_path: str = None, fps: int = 2, dpi: int = 100):
        """
        Generate animated GIF showing loop closure detection from predictions.
        
        Args:
            output_path: Path to save GIF (default: ./loop_closure_pred_{seq_name}.gif)
            fps: Frames per second in GIF
            dpi: DPI for image quality
            
        Returns:
            Path to saved GIF
        """
        if output_path is None:
            output_path = f"loop_closure_pred_{self.seq_name}_topk{self.topk}.gif"
        
        # Ensure output directory exists
        output_path = os.path.abspath(output_path)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating GIF: {output_path}")
        print(f"Total frames to visualize: {len(self.frame_indices)}")
        print(f"Top-K: {self.topk}")
        print(f"Distance threshold: {self.distance_threshold}m")
        
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


def load_predictions(prediction_file: str):
    """
    Load predictions from a pickle file.
    
    Args:
        prediction_file: Path to predictions.pkl file
        
    Returns:
        Dictionary of predictions
    """
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    
    with open(prediction_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded predictions from: {prediction_file}")
    print(f"  Number of queries: {len(predictions)}")
    
    return predictions


def generate_loop_closure_gifs_from_predictions(
    dataset_root_dir: str,
    prediction_root_dir: str,
    sequences: list,
    model_names: list,
    output_dir: str = None,
    topk: int = 1,
    distance_threshold: float = 10.0,
    sample_rate: int = 10
):
    """
    Generate loop closure GIFs for multiple sequences using model predictions.
    
    Args:
        dataset_root_dir: Root dataset directory (where sequences are located)
        prediction_root_dir: Root directory containing predictions (e.g., saved/hortov2/)
        sequences: List of sequence names
        model_names: List of model names (subdirectories in prediction_root_dir)
        output_dir: Directory to save GIFs (default: prediction_root_dir)
        topk: Top-K predictions to visualize
        distance_threshold: Distance threshold for valid loop closures (meters)
        sample_rate: Skip every N frames when sampling for visualization
    """
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(prediction_root_dir, 'loop_closure_gifs_predictions')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"GIFs will be saved to: {output_dir}")
    
    for seq in sequences:
        for model_name in model_names:
            print(f"\n{'='*80}")
            print(f"Processing: {seq} | Model: {model_name}")
            print(f"{'='*80}")
            
            try:
                # Load dataset
                fs = file_structure(dataset_root_dir, seq, verbose=False)
                
                print(f"Dataset loaded:")
                print(f"  Total positions: {len(fs._get_positions_())}")
                print(f"  Unique labels: {np.unique(fs._get_labels())}")
                
                # Find prediction file
                # Typical structure: prediction_root_dir/seq/model_name/predictions/place/<metric>@<K>/predictions.pkl
                pred_base_dir = os.path.join(prediction_root_dir, seq, model_name, 'predictions', 'place')
                
                if not os.path.exists(pred_base_dir):
                    print(f"  ✗ Predictions directory not found: {pred_base_dir}")
                    continue
                
                # Find the first predictions.pkl file in subdirectories
                pred_file = None
                for root, dirs, files in os.walk(pred_base_dir):
                    if 'predictions.pkl' in files:
                        pred_file = os.path.join(root, 'predictions.pkl')
                        break
                
                if pred_file is None:
                    print(f"  ✗ No predictions.pkl found in: {pred_base_dir}")
                    continue
                
                # Load predictions
                predictions = load_predictions(pred_file)
                
                # Create visualizer and generate GIF
                visualizer = LoopClosureVisualizerPredictions(
                    fs, predictions, f"{seq}_{model_name}",
                    topk=topk,
                    distance_threshold=distance_threshold,
                    sample_rate=sample_rate
                )
                
                output_filename = f"loop_closure_pred_{seq}_{model_name}_topk{topk}.gif"
                output_path = os.path.join(output_dir, output_filename)
                visualizer.generate_gif(output_path, fps=2, dpi=80)
                
                print(f"✓ Successfully generated GIF for {seq} | {model_name}")
                print(f"  Saved to: {output_path}\n")
                
            except Exception as e:
                import traceback
                print(f"✗ Error processing {seq} | {model_name}: {e}")
                traceback.print_exc()
                print()
                continue


if __name__ == '__main__':
    # Configuration
    DATASET_ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"
    PRED_ROOT_DIR = "/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2"
    OUTPUT_DIR = PRED_ROOT_DIR  # Where to save GIFs
    
    SEQUENCES = ["PCD_EASY", "PCD_Easy_DARK", "PCD_MED", "PCD_RAS_EASY"]
    
    # Model names (directories in saved/hortov2/SEQUENCE/)
    MODEL_NAMES = [
        "PointNetPGAP-None",
        "PointNetVLAD-None",
        "SPVSoAP3D-SoAP-log-pnl-fc-None",
        "LOGG3D-None",
        "overlap_transformer-None"
    ]
    
    # Parameters for prediction-based visualization
    TOPK = 1  # Show top-1 predictions
    DISTANCE_THRESHOLD = 10.0  # Maximum distance for valid loop closures (meters)
    
    # Visualization parameter - skip every N frames
    SAMPLE_RATE = 10  # Show every 10th query frame in the GIF
    
    print("Loop Closure Detection GIF Generator (Using PREDICTIONS)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Dataset root directory: {DATASET_ROOT_DIR}")
    print(f"  Prediction root directory: {PRED_ROOT_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sequences: {SEQUENCES}")
    print(f"  Models: {MODEL_NAMES}")
    print(f"  Top-K: {TOPK}")
    print(f"  Distance threshold: {DISTANCE_THRESHOLD} meters")
    print(f"  Sample rate: {SAMPLE_RATE} (show every {SAMPLE_RATE}th query frame)")
    print("=" * 80)
    
    # Generate GIFs
    generate_loop_closure_gifs_from_predictions(
        DATASET_ROOT_DIR,
        PRED_ROOT_DIR,
        SEQUENCES,
        MODEL_NAMES,
        output_dir=OUTPUT_DIR,
        topk=TOPK,
        distance_threshold=DISTANCE_THRESHOLD,
        sample_rate=SAMPLE_RATE
    )
    
    print("\n" + "=" * 80)
    print("All GIFs generated successfully!")
    print("=" * 80)
