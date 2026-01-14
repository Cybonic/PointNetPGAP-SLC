"""
Plot comparison of true positive loop closures across multiple models.

This script creates a multi-panel visualization showing true positive loop closures
for different place recognition models, similar to the ground truth visualization.
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


def load_predictions(predictions_path):
    """Load predictions from pickle file."""
    with open(predictions_path, 'rb') as f:
        return pickle.load(f)


def collect_true_positives(fs, predictions, topk=1, distance_threshold=10.0, 
                           min_temporal_distance=50):
    """
    Collect all true positive loop closures from predictions.
    
    Args:
        fs: file_structure object
        predictions: Dictionary of predictions
        topk: Top-K predictions to consider
        distance_threshold: Maximum distance for valid loop closures
        min_temporal_distance: Minimum frame distance to consider as loop closure
        
    Returns:
        List of tuples: (query_idx, neighbor_idx, distance)
    """
    true_positives = []
    
    for query_idx in sorted(predictions.keys()):
        pred_data = predictions[query_idx]
        pred_loops = pred_data['pred_loops']
        query_segment = pred_data['segment']
        
        # Get top-k predictions
        pred_indices = pred_loops['idx'][:topk]
        pred_distances = pred_loops['dist'][:topk]
        pred_segments = pred_loops['segment'][:topk]
        
        # Filter by distance threshold
        valid_mask = pred_distances <= distance_threshold
        valid_indices = pred_indices[valid_mask]
        valid_distances = pred_distances[valid_mask]
        valid_segments = pred_segments[valid_mask]
        
        # Filter by temporal distance
        temporal_distances = np.abs(query_idx - valid_indices)
        temporal_mask = temporal_distances >= min_temporal_distance
        valid_indices = valid_indices[temporal_mask]
        valid_distances = valid_distances[temporal_mask]
        valid_segments = valid_segments[temporal_mask]
        
        # Identify true positives (same segment)
        is_tp = (valid_segments == query_segment)
        
        for neighbor_idx, distance, is_positive in zip(valid_indices, valid_distances, is_tp):
            if is_positive:
                true_positives.append((int(query_idx), int(neighbor_idx), float(distance)))
    
    return true_positives


def plot_single_model(ax, fs, true_positives, title, 
                     connection_alpha=0.8, connection_linewidth=1.5,
                     show_points=False, view_angle=(30, 45)):
    """
    Plot true positives for a single model in a subplot.
    
    Args:
        ax: 3D axis object
        fs: file_structure object
        true_positives: List of (query_idx, neighbor_idx, distance) tuples
        title: Title for the subplot
        connection_alpha: Alpha for connection lines
        connection_linewidth: Width of connection lines
        show_points: Whether to highlight query/neighbor points
        view_angle: 3D view angle (elevation, azimuth)
    """
    # Get positions
    positions = fs._get_positions_()
    labels = fs._get_labels()
    
    # Align and elevate
    aligned_positions = aligned_path(positions)
    elevated_positions = elevate_along_path(aligned_positions, max_elevation=20.0)
    
    # Plot trajectory line in BLACK (prominent)
    ax.plot(elevated_positions[:, 0],
           elevated_positions[:, 1],
           elevated_positions[:, 2],
           'k-', alpha=1.0, linewidth=2.0, zorder=1)
    
    # Plot all true positive connections in GREEN
    for query_idx, neighbor_idx, distance in true_positives:
        query_pos = elevated_positions[query_idx]
        neighbor_pos = elevated_positions[neighbor_idx]
        
        # Draw connection line in GREEN
        ax.plot([query_pos[0], neighbor_pos[0]],
               [query_pos[1], neighbor_pos[1]],
               [query_pos[2], neighbor_pos[2]],
               'g-', alpha=connection_alpha, linewidth=connection_linewidth, zorder=10)
    
    # Set labels with smaller font
    ax.set_xlabel("X (m)", fontsize=10, labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=10, labelpad=8)
    ax.set_zlabel("Z (m)", fontsize=10, labelpad=8)
    
    # Set title
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Set equal aspect ratio
    max_range = np.array([
        elevated_positions[:, 0].max() - elevated_positions[:, 0].min(),
        elevated_positions[:, 1].max() - elevated_positions[:, 1].min(),
        elevated_positions[:, 2].max() - elevated_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (elevated_positions[:, 0].max() + elevated_positions[:, 0].min()) * 0.5
    mid_y = (elevated_positions[:, 1].max() + elevated_positions[:, 1].min()) * 0.5
    mid_z = (elevated_positions[:, 2].max() + elevated_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Adjust tick label size
    ax.tick_params(labelsize=8)


def plot_model_comparison(dataset_root, sequence, model_configs, output_path,
                         topk=1, distance_threshold=10.0, min_temporal_distance=50,
                         figsize=(24, 6), view_angle=(30, 45)):
    """
    Create a multi-panel comparison of true positives across models.
    
    Args:
        dataset_root: Path to dataset root
        sequence: Sequence name
        model_configs: List of (model_name, predictions_path) tuples
        output_path: Path to save the figure
        topk: Top-K predictions to consider
        distance_threshold: Maximum distance threshold
        min_temporal_distance: Minimum temporal distance
        figsize: Figure size (width, height)
        view_angle: 3D view angle (elevation, azimuth)
    """
    # Load dataset
    print(f"\nLoading dataset: {sequence}")
    fs = file_structure(dataset_root, sequence)
    print(f"Dataset loaded: {len(fs._get_positions_())} frames")
    
    # Determine grid layout
    n_models = len(model_configs)
    n_cols = min(5, n_models)  # Max 5 columns
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Adjust figure size based on layout
    fig_width = figsize[0]
    fig_height = figsize[1] * n_rows
    
    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    print("\n" + "=" * 80)
    print(f"Generating comparison for {n_models} models")
    print("=" * 80)
    
    # Process each model
    for idx, (model_name, pred_path) in enumerate(model_configs):
        print(f"\n[{idx+1}/{n_models}] Processing {model_name}...")
        
        # Load predictions
        if not os.path.exists(pred_path):
            print(f"  WARNING: Predictions not found at {pred_path}")
            continue
            
        predictions = load_predictions(pred_path)
        print(f"  Loaded {len(predictions)} predictions")
        
        # Collect true positives
        true_positives = collect_true_positives(
            fs, predictions, topk, distance_threshold, min_temporal_distance
        )
        print(f"  Found {len(true_positives)} true positives")
        
        # Create subplot
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
        
        # Plot
        plot_single_model(
            ax, fs, true_positives,
            title=f"{model_name}\nTP: {len(true_positives)}",
            connection_alpha=0.8,  # More visible green lines
            connection_linewidth=1.5,  # Thicker green lines
            show_points=False,
            view_angle=view_angle
        )
    
    # Overall title
    fig.suptitle(f"{sequence} - True Positive Loop Closures Comparison\n"
                f"Top-{topk} | Distance ≤ {distance_threshold}m | Temporal Gap ≥ {min_temporal_distance} frames",
                fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)
    print(f"\n{'=' * 80}")
    print(f"Figure saved to: {output_path}")
    print("=" * 80)
    
    plt.close(fig)


def main():
    """Main function for generating model comparison."""
    
    # Configuration
    dataset_root = "/home/tiago/workspace/place_uk/dataset/place_v2/PlaceRecognitionTestPolyTunnel"
    saved_root = "/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2"
    output_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives"
    
    # Sequence to visualize
    sequence = "PCD_MED"
    
    # Models to compare
    models = [
        "PointNetPGAP",
        "PointNetVLAD",
        "SPVSoAP3D",
        "LOGG3D",
        "overlap_transformer"
    ]
    
    # Build model configurations
    model_configs = []
    for model in models:
        # Find the predictions file in the appropriate subdirectory
        # Handle special naming convention for SPVSoAP3D
        if model == "SPVSoAP3D":
            model_dir_name = "SPVSoAP3D-SoAP-log-pnl-fc-None"
        else:
            model_dir_name = f"{model}-None"
            
        model_dir = os.path.join(saved_root, sequence, model_dir_name, "predictions", "place")
        
        # Find the recall@1 directory (format: X.XXX@1)
        if os.path.exists(model_dir):
            subdirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and '@1' in d]
            if subdirs:
                pred_path = os.path.join(model_dir, subdirs[0], "predictions.pkl")
                model_configs.append((model, pred_path))
            else:
                print(f"WARNING: No @1 directory found for {model}")
        else:
            print(f"WARNING: Model directory not found for {model}")
    
    # Parameters
    topk = 1
    distance_threshold = 10.0
    min_temporal_distance = 50
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("True Positive Loop Closures - Model Comparison")
    print("=" * 80)
    print(f"Dataset: {dataset_root}")
    print(f"Sequence: {sequence}")
    print(f"Models: {', '.join(models)}")
    print(f"Top-K: {topk}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Min temporal distance: {min_temporal_distance} frames")
    
    # Generate comparison
    output_path = os.path.join(output_dir, f"{sequence}_all_models_tp_comparison.png")
    
    plot_model_comparison(
        dataset_root=dataset_root,
        sequence=sequence,
        model_configs=model_configs,
        output_path=output_path,
        topk=topk,
        distance_threshold=distance_threshold,
        min_temporal_distance=min_temporal_distance,
        figsize=(24, 6),
        view_angle=(30, 45)
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
