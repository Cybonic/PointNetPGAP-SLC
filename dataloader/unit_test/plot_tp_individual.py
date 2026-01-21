"""
Generate individual true positive loop closure plots for each model and sequence.

This script creates separate visualizations for every model-sequence combination with:
- Consistent viewpoints across all plots
- No background grid
- Black trajectory paths
- Green loop closure connections
- Clean, publication-ready output
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle
import argparse
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path


def crop_image_tight(image_path, extra_margin=0):
    """
    Crop an image tightly around non-transparent content.
    
    Args:
        image_path: Path to the image file
        extra_margin: Extra pixels to keep around the content (can be negative to crop more)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGBA if not already
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Get the bounding box of non-transparent content
        alpha = img.split()[-1]  # Get alpha channel
        bbox = alpha.getbbox()
        
        if bbox:
            # Apply extra margin
            left, top, right, bottom = bbox
            left = max(0, left - extra_margin)
            top = max(0, top - extra_margin)
            right = min(img.width, right + extra_margin)
            bottom = min(img.height, bottom + extra_margin)
            
            # Crop to content
            img_cropped = img.crop((left, top, right, bottom))
            
            # Save back to the same file
            if image_path.endswith('.png'):
                img_cropped.save(image_path, format='PNG', dpi=(300, 300))
            elif image_path.endswith('.pdf'):
                # For PDF, convert to RGB first
                if img_cropped.mode == 'RGBA':
                    # Create white background
                    bg = Image.new('RGB', img_cropped.size, (255, 255, 255))
                    bg.paste(img_cropped, mask=img_cropped.split()[-1])
                    img_cropped = bg
                img_cropped.save(image_path, format='PDF', resolution=300)
            
            return True
    except Exception as e:
        print(f"    Warning: Could not crop {image_path}: {e}")
        return False
    
    return False


def elevate_path_by_revisits(positions, spatial_threshold=10.0, level_height=5.0, min_temporal_gap=50, transition_length=20):
    """
    Elevate path based on the number of times it revisits the same spatial location.
    Each revisit gets a fixed vertical offset. Levels never decrease (monotonic increase).
    Smooth transitions between levels.
    
    Args:
        positions: Nx3 array of (x, y, z) positions
        spatial_threshold: Distance threshold to consider as "same location" (meters)
        level_height: Height offset for each revisit level (meters)
        min_temporal_gap: Minimum frame gap to consider as separate visit (frames)
        transition_length: Number of points over which to smooth level transitions
        
    Returns:
        Nx3 array with elevated z-coordinates based on revisit levels, and smoothed levels array
    """
    elevated_positions = positions.copy()
    n_points = len(positions)
    
    # Track which level each point should be at
    levels = np.zeros(n_points, dtype=int)
    current_level = 0  # Track the current level (never decreases)
    
    # For each point, determine its level based on previous visits to nearby locations
    for i in range(n_points):
        current_pos = positions[i, :2]  # Only x, y for spatial comparison
        
        # Find all previous points within spatial threshold
        # BUT only check points that are temporally separated (not part of continuous path)
        if i > min_temporal_gap:
            # Only check points that are at least min_temporal_gap frames back
            prev_positions = positions[:i-min_temporal_gap, :2]
            distances = np.linalg.norm(prev_positions - current_pos, axis=1)
            nearby_mask = distances < spatial_threshold
            
            if np.any(nearby_mask):
                # Get the levels of nearby previous points
                nearby_levels = levels[:i-min_temporal_gap][nearby_mask]
                # Should be one level higher than the max nearby level
                suggested_level = np.max(nearby_levels) + 1
                # But never decrease from current level
                current_level = max(current_level, suggested_level)
        
        levels[i] = current_level
    
    # Create smooth transitions between levels
    smooth_levels = levels.astype(float).copy()
    
    # Find level change points
    for i in range(1, n_points):
        if levels[i] > levels[i-1]:
            # Level increased - create smooth transition
            level_change = levels[i] - levels[i-1]
            start_idx = max(0, i - transition_length // 2)
            end_idx = min(n_points, i + transition_length // 2)
            transition_range = end_idx - start_idx
            
            if transition_range > 0:
                # Linear interpolation for smooth transition
                for j in range(start_idx, end_idx):
                    # Don't go below the level at start_idx or above level at end_idx
                    progress = (j - start_idx) / transition_range
                    base_level = levels[start_idx]
                    smooth_levels[j] = base_level + level_change * progress
                    # Ensure monotonic increase
                    if j > 0:
                        smooth_levels[j] = max(smooth_levels[j], smooth_levels[j-1])
    
    # Apply smooth elevation based on smoothed levels
    elevated_positions[:, 2] = positions[:, 2] + smooth_levels * level_height
    
    return elevated_positions, levels  # Return original discrete levels for reporting



def load_predictions(predictions_path):
    """
    Load predictions from pickle file.
    
    Handles different file formats:
    1. Direct predictions dict: {query_idx: prediction_data, ...}
    2. Wrapped format: {'predictions': {query_idx: prediction_data, ...}, ...}
    
    Returns:
        Dictionary of predictions keyed by query_idx, or None if file not found
    """
    if not os.path.exists(predictions_path):
        return None
    with open(predictions_path, 'rb') as f:
        data = pickle.load(f)
    
    # Check if predictions are wrapped in a 'predictions' key
    if isinstance(data, dict) and 'predictions' in data:
        # Wrapped format (e.g., ScanContext output)
        return data['predictions']
    else:
        # Direct predictions dict
        return data


def find_predictions_path(saved_root, sequence, model):
    """
    Find the predictions.pkl file for a given model and sequence.
    
    Handles different file structures:
    1. Standard models: {model}-None/predictions/place/{recall}@1/predictions.pkl
    2. ScanContext: ScanContext/predictions.pkl (directly in model folder)
    
    Args:
        saved_root: Root directory for saved predictions
        sequence: Sequence name
        model: Model name
        
    Returns:
        Tuple of (predictions_path, model_dir_name) or (None, None) if not found
    """
    # Handle special naming conventions
    if model == "SPVSoAP3D":
        model_dir_name = "SPVSoAP3D-SoAP-log-pnl-fc-None"
    elif model == "ScanContext":
        model_dir_name = "ScanContext"  # No -None suffix
    else:
        model_dir_name = f"{model}-None"
    
    model_base_dir = os.path.join(saved_root, sequence, model_dir_name)
    
    if not os.path.exists(model_base_dir):
        return None, model_dir_name
    
    # Try ScanContext style first (predictions.pkl directly in model folder)
    direct_pred_path = os.path.join(model_base_dir, "predictions.pkl")
    if os.path.exists(direct_pred_path):
        return direct_pred_path, model_dir_name
    
    # Try standard style: predictions/place/{recall}@1/predictions.pkl
    place_dir = os.path.join(model_base_dir, "predictions", "place")
    if os.path.exists(place_dir):
        # Find the recall@1 directory
        subdirs = [d for d in os.listdir(place_dir) 
                  if os.path.isdir(os.path.join(place_dir, d)) and '@1' in d]
        
        if subdirs:
            pred_path = os.path.join(place_dir, subdirs[0], "predictions.pkl")
            if os.path.exists(pred_path):
                return pred_path, model_dir_name
    
    return None, model_dir_name


def detect_prediction_format(pred_data):
    """
    Detect the format of prediction data.
    
    Returns:
        'new' if using new format (candidates, positions_dist, labels, query_label)
        'old' if using old format (pred_loops with idx/dist/segment, segment)
    """
    if 'candidates' in pred_data and 'positions_dist' in pred_data:
        return 'new'
    elif 'pred_loops' in pred_data:
        return 'old'
    else:
        raise ValueError(f"Unknown prediction format. Keys: {pred_data.keys()}")


def extract_prediction_data(pred_data, topk):
    """
    Extract prediction data in a unified format regardless of input format.
    
    Args:
        pred_data: Dictionary containing prediction data
        topk: Number of top predictions to extract
        
    Returns:
        Tuple of (pred_indices, pred_distances, pred_segments, query_segment)
        All as numpy arrays
    """
    format_type = detect_prediction_format(pred_data)
    
    if format_type == 'new':
        # New format: candidates, positions_dist, labels, query_label
        pred_indices = np.array(pred_data['candidates'][:topk])
        pred_distances = np.array(pred_data['positions_dist'][:topk])
        pred_segments = np.array(pred_data['labels'][:topk])
        query_segment = pred_data['query_label']
    else:
        # Old format: pred_loops with idx/dist/segment, segment
        pred_loops = pred_data['pred_loops']
        pred_indices = np.array(pred_loops['idx'][:topk])
        pred_distances = np.array(pred_loops['dist'][:topk])
        pred_segments = np.array(pred_loops['segment'][:topk])
        query_segment = pred_data['segment']
    
    return pred_indices, pred_distances, pred_segments, query_segment


def collect_true_positives(predictions, topk=1, distance_threshold=10.0, 
                           min_temporal_distance=50):
    """
    Collect all true positive loop closures from predictions.
    
    Supports both old and new prediction formats:
    - Old format: pred_loops with idx/dist/segment keys, segment for query label
    - New format: candidates, positions_dist, labels, query_label
    
    IMPORTANT RETRIEVAL RULES:
    1. Retrieval is ALWAYS done in PAST frames only (never future frames)
    2. The nearest neighbor is the CLOSEST point, even if it has been retrieved before
    3. The same past frame can be the nearest neighbor for multiple query frames
    
    Args:
        predictions: Dictionary of predictions (keyed by query_idx)
        topk: Top-K predictions to consider
        distance_threshold: Maximum distance for valid loop closures
        min_temporal_distance: Minimum frame distance to consider as loop closure
        
    Returns:
        List of tuples: (query_idx, neighbor_idx, distance)
    """
    true_positives = []
    
    # Detect format from first prediction
    first_key = next(iter(predictions.keys()))
    format_type = detect_prediction_format(predictions[first_key])
    print(f"  Detected prediction format: {format_type}")
    
    for query_idx in sorted(predictions.keys()):
        pred_data = predictions[query_idx]
        
        # Extract data in unified format
        pred_indices, pred_distances, pred_segments, query_segment = extract_prediction_data(pred_data, topk)
        
        # Skip if no predictions
        if len(pred_indices) == 0:
            continue
        
        # RETRIEVAL RULE: Filter to keep only PAST frames (neighbor_idx < query_idx)
        # This ensures we NEVER retrieve from future frames
        past_mask = pred_indices < query_idx
        pred_indices = pred_indices[past_mask]
        pred_distances = pred_distances[past_mask]
        pred_segments = pred_segments[past_mask]
        
        if len(pred_indices) == 0:
            continue
        
        # Filter by distance threshold
        valid_mask = pred_distances <= distance_threshold
        valid_indices = pred_indices[valid_mask]
        valid_distances = pred_distances[valid_mask]
        valid_segments = pred_segments[valid_mask]
        
        if len(valid_indices) == 0:
            continue
        
        # Filter by temporal distance
        temporal_distances = query_idx - valid_indices  # Now always positive since valid_indices < query_idx
        temporal_mask = temporal_distances >= min_temporal_distance
        valid_indices = valid_indices[temporal_mask]
        valid_distances = valid_distances[temporal_mask]
        valid_segments = valid_segments[temporal_mask]
        
        if len(valid_indices) == 0:
            continue
        
        # Identify true positives (same segment)
        is_tp = (valid_segments == query_segment)
        
        # Add true positives
        # NOTE: We don't exclude neighbors that have been retrieved before
        # The CLOSEST point is always selected, regardless of previous retrievals
        for neighbor_idx, distance, is_positive in zip(valid_indices, valid_distances, is_tp):
            if is_positive:
                true_positives.append((int(query_idx), int(neighbor_idx), float(distance)))
    
    return true_positives


def collect_ground_truth_loops(fs, distance_threshold=10.0, min_temporal_distance=50):
    """
    Collect all ground truth loop closures based on segment labels.
    
    IMPORTANT RETRIEVAL RULES:
    1. Retrieval is ALWAYS done in PAST frames only (never future frames)
    2. The nearest neighbor is the CLOSEST point, even if it has been retrieved before
    3. The same past frame can be the nearest neighbor for multiple query frames
    
    Args:
        fs: file_structure object
        distance_threshold: Maximum distance for valid loop closures
        min_temporal_distance: Minimum frame distance to consider as loop closure
        
    Returns:
        List of tuples: (query_idx, neighbor_idx, distance)
    """
    # Use the dataset's ground truth loop closure function
    gt_dict = fs.get_ground_truth_loop_closure(
        warm_up=100, 
        lower_bound_idx=min_temporal_distance, 
        distance_threshold=distance_threshold, 
        topk=1
    )
    
    # Convert from dictionary format to list of tuples format
    ground_truth_loops = []
    query_indices = gt_dict['query_indices']
    neighbor_indices = gt_dict['neighbor_indices']
    distances = gt_dict['distances']
    
    for query_idx, neighbor_idx, distance in zip(query_indices, neighbor_indices, distances):
        ground_truth_loops.append((int(query_idx), int(neighbor_idx), float(distance)))
    
    return ground_truth_loops


def plot_model_sequence(fs, true_positives, model_name, sequence, output_path,
                        view_elev=30, view_azim=45, figsize=(16, 12),
                        connection_alpha=0.8, connection_linewidth=2.0,
                        show_grid=False, show_legend=True, show_axes=True,
                        spatial_threshold=10.0, level_height=5.0, min_temporal_gap=50,
                        subsample_factor=5, transition_length=20, tight_crop=True,
                        crop_margin=10):
    """
    Create a single plot for one model-sequence combination.
    Saves in both PDF (vector) and PNG (raster) formats.
    
    Args:
        fs: file_structure object
        true_positives: List of (query_idx, neighbor_idx, distance) tuples
        model_name: Name of the model
        sequence: Sequence name
        output_path: Path to save the figure (PDF format, PNG will be saved alongside)
        view_elev: Elevation angle for 3D view
        view_azim: Azimuth angle for 3D view
        figsize: Figure size (width, height)
        connection_alpha: Alpha for connection lines
        connection_linewidth: Width of connection lines
        show_grid: Whether to show grid
        show_legend: Whether to show legend
        show_axes: Whether to show axis labels
        spatial_threshold: Distance threshold for considering same location (meters)
        level_height: Height offset for each revisit level (meters)
        min_temporal_gap: Minimum frame gap to consider as separate visit (frames)
        subsample_factor: Show every Nth loop closure prediction (e.g., 5 = show every 5th prediction)
        transition_length: Number of points over which to smooth level transitions
        tight_crop: Whether to crop images tightly around content
        crop_margin: Extra pixels to keep around content when cropping (negative to crop more)
    """
    # Get positions
    positions = fs._get_positions_()
    
    # Align positions
    aligned_positions = aligned_path(positions)
    
    # Elevate based on revisits to same location with smooth transitions
    elevated_positions, levels = elevate_path_by_revisits(
        aligned_positions, 
        spatial_threshold=spatial_threshold, 
        level_height=level_height,
        min_temporal_gap=min_temporal_gap,
        transition_length=transition_length
    )
    
    print(f"    Elevation levels: {np.unique(levels)} (max level: {np.max(levels)})")

    
    # Subsample the predictions (loop closures), not the path
    if subsample_factor > 1:
        subsampled_tp = true_positives[::subsample_factor]
        print(f"    Subsampled predictions: {len(true_positives)} -> {len(subsampled_tp)} loop closures (factor: {subsample_factor})")
    else:
        subsampled_tp = true_positives

    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot FULL trajectory in BLACK (monocolor)
    ax.plot(elevated_positions[:, 0],
           elevated_positions[:, 1],
           elevated_positions[:, 2],
           'k-', alpha=1.0, linewidth=2.5, zorder=1)
    
    # Plot SUBSAMPLED true positive connections in GREEN
    connection_plotted = False
    for query_idx, neighbor_idx, distance in subsampled_tp:
        query_pos = elevated_positions[query_idx]
        neighbor_pos = elevated_positions[neighbor_idx]
        
        # Draw connection line in GREEN (label only first one for legend)
        if show_legend and not connection_plotted:
            ax.plot([query_pos[0], neighbor_pos[0]],
                   [query_pos[1], neighbor_pos[1]],
                   [query_pos[2], neighbor_pos[2]],
                   'g-', alpha=connection_alpha, linewidth=connection_linewidth, 
                   zorder=10, label=f'Loop Closures ({len(subsampled_tp)}/{len(true_positives)} shown)')
            connection_plotted = True
        else:
            ax.plot([query_pos[0], neighbor_pos[0]],
                   [query_pos[1], neighbor_pos[1]],
                   [query_pos[2], neighbor_pos[2]],
                   'g-', alpha=connection_alpha, linewidth=connection_linewidth, zorder=10)
    
    # Remove all axes elements
    ax.set_axis_off()
    
    # Optionally keep title if show_legend is True
    if show_legend:
        title = f"{model_name} - {sequence}\n"
        title += f"True Positives: {len(true_positives)}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)
    
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
    
    # Remove background elements and make transparent
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)
    
    # Make figure background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    # Save figure as PDF with transparent background and tight bounding box
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0, 
                transparent=True, facecolor='none')
    
    # Also save as PNG with high resolution
    png_path = output_path.replace('.pdf', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0, 
                transparent=True, facecolor='none', dpi=300)
    
    plt.close(fig)
    
    # Crop images tightly if requested
    if tight_crop:
        # Crop PNG
        if crop_image_tight(png_path, extra_margin=crop_margin):
            pass  # Success message handled by caller
        
        # Note: PDF cropping is more complex and may not work as well
        # For now, we'll just crop the PNG which is what's used in the combined grid


def generate_all_plots(dataset_root, saved_root, sequences, models, output_dir,
                      topk=1, distance_threshold=10.0, min_temporal_distance=50,
                      view_elev=30, view_azim=45, figsize=(16, 12),
                      show_grid=False, show_legend=True, show_axes=True,
                      spatial_threshold=10.0, level_height=5.0, min_temporal_gap=50,
                      subsample_factor=5, transition_length=20, tight_crop=True,
                      crop_margin=10):
    """
    Generate individual plots for all model-sequence combinations.
    
    Args:
        dataset_root: Path to dataset root
        saved_root: Root directory for saved predictions
        sequences: List of sequence names
        models: List of model names
        output_dir: Directory to save plots
        topk: Top-K predictions to consider
        distance_threshold: Maximum distance threshold
        min_temporal_distance: Minimum temporal distance
        view_elev: Elevation angle for all plots
        view_azim: Azimuth angle for all plots
        figsize: Figure size for all plots
        show_grid: Whether to show grid
        show_legend: Whether to show legend
        show_axes: Whether to show axis labels
        spatial_threshold: Distance threshold for considering same location (meters)
        level_height: Height offset for each revisit level (meters)
        min_temporal_gap: Minimum frame gap to consider as separate visit (frames)
        subsample_factor: Show every Nth loop closure prediction (e.g., 5 = show every 5th prediction)
        transition_length: Number of points over which to smooth level transitions
        tight_crop: Whether to crop images tightly around content
        crop_margin: Extra pixels to keep around content when cropping
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Generating Individual True Positive Plots")
    print("=" * 80)
    print(f"Sequences: {sequences}")
    print(f"Models: {models}")
    print(f"Output directory: {output_dir}")
    print(f"View angle: elevation={view_elev}°, azimuth={view_azim}°")
    print(f"Grid: {'ON' if show_grid else 'OFF'}")
    print(f"Top-K: {topk}")
    print(f"Distance threshold: {distance_threshold}m")
    print(f"Min temporal distance: {min_temporal_distance} frames")
    print(f"Elevation: spatial_threshold={spatial_threshold}m, level_height={level_height}m, min_temporal_gap={min_temporal_gap} frames")
    print(f"Tight crop: {'ENABLED' if tight_crop else 'DISABLED'} (margin: {crop_margin}px)")
    print("=" * 80)
    
    total_plots = len(sequences) * len(models) + len(sequences)  # +1 ground truth per sequence
    plot_count = 0
    
    results_summary = []
    
    for sequence in sequences:
        print(f"\n{'=' * 80}")
        print(f"Processing Sequence: {sequence}")
        print(f"{'=' * 80}")
        
        # Load dataset once per sequence
        try:
            fs = file_structure(dataset_root, sequence)
            print(f"Dataset loaded: {len(fs._get_positions_())} frames")
        except Exception as e:
            print(f"ERROR loading dataset for {sequence}: {e}")
            continue
        
        # Generate ground truth plot first
        plot_count += 1
        print(f"\n[{plot_count}/{total_plots}] Ground Truth - {sequence}")
        print("-" * 80)
        
        # Collect ground truth loop closures
        ground_truth_loops = collect_ground_truth_loops(
            fs, distance_threshold, min_temporal_distance
        )
        print(f"  Found {len(ground_truth_loops)} ground truth loop closures")
        
        # Subsample ground truth for visualization
        if subsample_factor > 1:
            subsample_factor_gt = subsample_factor 
            subsampled_gt = ground_truth_loops[::subsample_factor_gt]
            print(f"  Subsampled: {len(ground_truth_loops)} -> {len(subsampled_gt)} loop closures (factor: {subsample_factor_gt})")
        else:
            subsampled_gt = ground_truth_loops
        
        # Generate ground truth plot
        gt_output_path = os.path.join(output_dir, f"{sequence}_GroundTruth_loops.pdf")
        
        plot_model_sequence(
            fs=fs,
            true_positives=subsampled_gt,
            model_name="Ground Truth",
            sequence=sequence,
            output_path=gt_output_path,
            view_elev=view_elev,
            view_azim=view_azim,
            figsize=figsize,
            show_grid=show_grid,
            show_legend=show_legend,
            show_axes=show_axes,
            spatial_threshold=spatial_threshold,
            level_height=level_height,
            min_temporal_gap=min_temporal_gap,
            subsample_factor=1,  # Already subsampled
            transition_length=transition_length,
            tight_crop=tight_crop,
            crop_margin=crop_margin
        )
        
        print(f"  ✓ Saved: {gt_output_path}")
        print(f"  ✓ Saved: {gt_output_path.replace('.pdf', '.png')}")
        results_summary.append((sequence, "Ground Truth", "SUCCESS", len(ground_truth_loops)))
        
        # Continue with model predictions
        for model in models:
            plot_count += 1
            print(f"\n[{plot_count}/{total_plots}] {model} - {sequence}")
            print("-" * 80)
            
            # Find predictions file (handles different file structures)
            pred_path, model_dir_name = find_predictions_path(saved_root, sequence, model)
            
            if pred_path is None:
                print(f"  WARNING: Predictions not found for {model} in {sequence}")
                print(f"    Searched in: {os.path.join(saved_root, sequence, model_dir_name)}")
                results_summary.append((sequence, model, "NOT_FOUND", 0))
                continue
            
            print(f"  Found predictions at: {pred_path}")
            
            # Load predictions
            predictions = load_predictions(pred_path)
            if predictions is None:
                print(f"  WARNING: Could not load predictions from {pred_path}")
                results_summary.append((sequence, model, "LOAD_FAILED", 0))
                continue
            
            print(f"  Loaded {len(predictions)} predictions")
            
            # Collect true positives
            true_positives = collect_true_positives(
                predictions, topk, distance_threshold, min_temporal_distance
            )
            print(f"  Found {len(true_positives)} true positives")
            
            # Generate plot - output as PDF
            output_path = os.path.join(output_dir, f"{sequence}_{model}_tp.pdf")
            
            plot_model_sequence(
                fs=fs,
                true_positives=true_positives,
                model_name=model,
                sequence=sequence,
                output_path=output_path,
                view_elev=view_elev,
                view_azim=view_azim,
                figsize=figsize,
                show_grid=show_grid,
                show_legend=show_legend,
                show_axes=show_axes,
                spatial_threshold=spatial_threshold,
                level_height=level_height,
                min_temporal_gap=min_temporal_gap,
                subsample_factor=subsample_factor,
                transition_length=transition_length,
                tight_crop=tight_crop,
                crop_margin=crop_margin
            )
            
            print(f"  ✓ Saved: {output_path}")
            print(f"  ✓ Saved: {output_path.replace('.pdf', '.png')}")
            results_summary.append((sequence, model, "SUCCESS", len(true_positives)))
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Sequence':<20} {'Model':<25} {'Status':<20} {'TPs':<10}")
    print("-" * 80)
    
    for seq, model, status, tp_count in results_summary:
        if status == "SUCCESS":
            print(f"{seq:<20} {model:<25} {status:<20} {tp_count:<10}")
        else:
            print(f"{seq:<20} {model:<25} {status:<20} {'N/A':<10}")
    
    success_count = sum(1 for _, _, status, _ in results_summary if status == "SUCCESS")
    print("-" * 80)
    print(f"Total successful plots: {success_count}/{total_plots}")
    print("=" * 80)


def main():
    """Main function with configurable parameters."""
    
    # ============================================================================
    # CONFIGURATION SECTION - EDIT THESE PARAMETERS
    # ============================================================================
    
    # Dataset and output paths
    dataset_root = "/home/tiago/workspace/place_uk/dataset/place_v2/PlaceRecognitionTestPolyTunnel"
    saved_root = "/home/tiago/workspace/place_uk/PointNetGAP/saved_v3/hortov2"
    output_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots/true_positives_individual"
    
    # Sequences to process
    sequences = [
        # "PCD_EASY",      # Uncomment if predictions available
        "PCD_Easy_DARK",
        "PCD_MED",
        # "PCD_RAS_EASY"   # Uncomment if dataset path is fixed
    ]
    
    # Models to process
    models = [
        "PointNetPGAP",
        "PointNetVLAD",
        "SPVSoAP3D",
        "LOGG3D",
        "overlap_transformer",
        "ScanContext",
    ]
    
    # Loop closure parameters
    topk = 1                      # Top-K predictions to consider
    distance_threshold = 10.0     # Maximum distance in meters
    min_temporal_distance = 50    # Minimum frame gap
    
    # Visualization parameters
    view_elev = 30               # Elevation angle (degrees)
    view_azim = 135               # Azimuth angle (degrees)
    figsize = (16, 12)           # Figure size (width, height)
    
    # Display options
    show_grid = False            # Show/hide background grid
    show_legend = False          # Show/hide legend
    show_axes = False            # Show/hide axis labels and ticks
    
    # Connection line appearance
    connection_alpha = 0.8       # Transparency of green lines (0-1)
    connection_linewidth = 2.0   # Thickness of green lines
    
    # Elevation parameters
    spatial_threshold = 10.0     # Distance threshold for same location (meters)
    level_height = 5.0           # Height offset per revisit level (meters)
    min_temporal_gap = 50        # Minimum frame gap to consider as separate visit (frames)
    
    # Transition smoothing
    transition_length = 20       # Number of points over which to smooth transitions (higher = smoother)
    
    # Subsampling parameter (for loop closures, not the path)
    subsample_factor = 2         # Show every Nth loop closure (1=all, 5=every 5th, 10=every 10th)
    
    # Tight cropping parameters
    tight_crop = True            # Enable tight cropping around content
    crop_margin = 10             # Extra pixels to keep around content (negative to crop more)
    
    # ============================================================================
    # END CONFIGURATION
    # ============================================================================
    
    # Generate all plots
    generate_all_plots(
        dataset_root=dataset_root,
        saved_root=saved_root,
        sequences=sequences,
        models=models,
        output_dir=output_dir,
        topk=topk,
        distance_threshold=distance_threshold,
        min_temporal_distance=min_temporal_distance,
        view_elev=view_elev,
        view_azim=view_azim,
        figsize=figsize,
        show_grid=show_grid,
        show_legend=show_legend,
        show_axes=show_axes,
        spatial_threshold=spatial_threshold,
        level_height=level_height,
        min_temporal_gap=min_temporal_gap,
        subsample_factor=subsample_factor,
        transition_length=transition_length,
        tight_crop=tight_crop,
        crop_margin=crop_margin
    )
    
    print("\nDone! All plots generated.")


if __name__ == "__main__":
    main()
