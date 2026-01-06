import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path

COLORS = generate_label_colors(200)
ROOT_DIR = os.path.abspath("/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel")
SEQs = ["PCD_EASY",
        "PCD_Easy_DARK",
        "PCD_MED",
        "PCD_RAS_EASY"]


def test_nearest_neighbors_different_frame():
    """Test nearest neighbor computation with different frame IDs."""
    
    for seq in SEQs:
        seq_dir = os.path.join(ROOT_DIR, seq)
        print(f"\n{'='*60}")
        print(f"Testing nearest neighbors for: {seq}")
        print(f"{'='*60}")
        
        fs = file_structure(ROOT_DIR, seq, verbose=True)
        
        # Get positions, labels, and frame IDs
        positions = fs._get_positions_()
        labels = fs._get_labels()
        frame_ids = fs._get_frame_ids()
        
        print(f"Total positions: {len(positions)}")
        print(f"Unique frame IDs: {np.unique(frame_ids)}")
        print(f"Unique labels: {np.unique(labels)}")
        print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"Frame ID distribution: {dict(zip(*np.unique(frame_ids, return_counts=True)))}")
        
        # Find nearest neighbor of position 50 with different frame ID
        if len(positions) > 50:
            result = fs.compute_nearest_neighbor_different_frame(50)  # Changed from compute_nearest_neighbor_label
            if result['neighbor_idx'] is not None:
                print(f"\nPosition 50 (label {result['query_label']}, frame {result['query_frame']}) has nearest neighbor at index {result['neighbor_idx']} (frame {result['neighbor_frame']}) at distance {result['distance']:.2f}m")
            else:
                print(f"\nPosition 50 (label {result['query_label']}, frame {result['query_frame']}) has no neighbors with different frame ID")

        # Compute for all positions
        print(f"\nComputing nearest neighbors for all positions...")
        all_neighbors = fs.compute_all_nearest_neighbors_different_frame()

        # Print statistics
        distances = [r['distance'] for r in all_neighbors if r['distance'] != np.inf]
        if distances:
            print(f"Nearest neighbor distances (different frame):")
            print(f"  Min: {np.min(distances):.2f}m")
            print(f"  Max: {np.max(distances):.2f}m")
            print(f"  Mean: {np.mean(distances):.2f}m")
            print(f"  Median: {np.median(distances):.2f}m")
        else:
            print(f"No neighbors found with different frame IDs")
        
        # Show first 10 neighbors
        print(f"\nFirst 10 nearest neighbors with different frame IDs:")
        shown = 0
        for i, result in enumerate(all_neighbors):
            if result['distance'] != np.inf and shown < 50:
                print(f"  Position {i}: frame {result['query_frame']} -> neighbor {result['neighbor_idx']} (frame {result['neighbor_frame']}) at {result['distance']:.2f}m")
                shown += 1
        
        # Plot 3D path with nearest neighbor connections
        plot_nearest_neighbors_3d(positions, all_neighbors, seq)


def plot_nearest_neighbors_3d(positions, neighbors, seq_name):
    """
    Plot 3D path with connections to nearest neighbors of different frame IDs.
    """
    # Align and elevate positions
    aligned_positions = aligned_path(positions)
    elevated_positions = elevate_along_path(aligned_positions, max_elevation=20.0)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot path line
    ax.plot(elevated_positions[:, 0], elevated_positions[:, 1], elevated_positions[:, 2],
            'k-', alpha=1, linewidth=5)
    
    # Plot connections to nearest neighbors (sample every Nth point for clarity)
    sample_rate = max(1, len(neighbors) // 50)  # Show ~50 connections
    for i in range(0, len(neighbors), sample_rate):
        neighbor = neighbors[i]
        if neighbor['distance'] != np.inf and neighbor['neighbor_idx'] is not None:
            query_pos = elevated_positions[i]
            nn_pos = elevated_positions[neighbor['neighbor_idx']]
            
            # Draw line from query point to nearest neighbor
            ax.plot([query_pos[0], nn_pos[0]], 
                   [query_pos[1], nn_pos[1]], 
                   [query_pos[2], nn_pos[2]],
                   'g-', alpha=0.7, linewidth=2.0)
            
            # Mark the connection endpoints with circles
            ax.scatter(*query_pos, color='red', s=50, alpha=0.6, marker='o', edgecolors='darkred', linewidth=1)
            ax.scatter(*nn_pos, color='orange', s=50, alpha=0.6, marker='s', edgecolors='darkorange', linewidth=1)
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (Elevation)")
    ax.set_title(f"3D Path with Nearest Neighbors (Different Frame IDs) - {seq_name}")
    
    # Keep equal aspect ratio
    max_range = np.array([elevated_positions[:, 0].max()-elevated_positions[:, 0].min(),
                          elevated_positions[:, 1].max()-elevated_positions[:, 1].min(),
                          elevated_positions[:, 2].max()-elevated_positions[:, 2].min()]).max() / 2.0
    
    mid_x = (elevated_positions[:, 0].max()+elevated_positions[:, 0].min()) * 0.5
    mid_y = (elevated_positions[:, 1].max()+elevated_positions[:, 1].min()) * 0.5
    mid_z = (elevated_positions[:, 2].max()+elevated_positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linestyle='--', linewidth=1.5, label='NN Connection (Different Frame)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Query Point'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=8, label='Nearest Neighbor')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Testing nearest neighbor computation with different frame IDs...")
    test_nearest_neighbors_different_frame()
    print("\nAll tests completed!")