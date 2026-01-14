"""
Example: Generate plots with different viewpoint configurations.

This script demonstrates how to generate the same plots with different
camera angles to find the best viewpoint for your data.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Import the main generation function
from plot_tp_individual import generate_all_plots

# Configuration
dataset_root = "/home/tiago/workspace/place_uk/dataset/place_v2/PlaceRecognitionTestPolyTunnel"
saved_root = "/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2"
base_output_dir = "/home/tiago/workspace/place_uk/PointNetGAP/plots/viewpoint_comparison"

sequences = ["PCD_MED"]  # Test with one sequence
models = ["PointNetPGAP", "PointNetVLAD"]  # Test with two models

# Define different viewpoints to test
viewpoint_configs = {
    "standard": (30, 45),      # Standard isometric view
    "top_down": (90, 0),       # Bird's eye view
    "high_angle": (60, 45),    # High elevation
    "side_view": (10, 0),      # Nearly horizontal
    "diagonal": (30, 60),      # More rotation
    "back_view": (30, 135),    # View from opposite side
}

print("=" * 80)
print("Generating Viewpoint Comparison")
print("=" * 80)
print(f"Testing {len(viewpoint_configs)} different viewpoints")
print(f"Sequences: {sequences}")
print(f"Models: {models}")
print("=" * 80)

for viewpoint_name, (elev, azim) in viewpoint_configs.items():
    print(f"\n{'=' * 80}")
    print(f"Viewpoint: {viewpoint_name} (elevation={elev}°, azimuth={azim}°)")
    print("=" * 80)
    
    output_dir = os.path.join(base_output_dir, viewpoint_name)
    
    generate_all_plots(
        dataset_root=dataset_root,
        saved_root=saved_root,
        sequences=sequences,
        models=models,
        output_dir=output_dir,
        topk=1,
        distance_threshold=10.0,
        min_temporal_distance=50,
        view_elev=elev,
        view_azim=azim,
        figsize=(16, 12),
        show_grid=False,
        show_legend=True,
        show_axes=True
    )

print("\n" + "=" * 80)
print("Viewpoint Comparison Complete!")
print("=" * 80)
print(f"\nOutput directory: {base_output_dir}")
print("\nGenerated viewpoints:")
for viewpoint_name, (elev, azim) in viewpoint_configs.items():
    print(f"  - {viewpoint_name}: elevation={elev}°, azimuth={azim}°")
print("\nCompare the plots to choose the best viewpoint for your publication.")
print("=" * 80)
