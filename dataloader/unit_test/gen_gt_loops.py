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
SEQs = [#"PCD_EASY",
        #"PCD_Easy_DARK",
        #"PCD_MED",
        "PCD_RAS_EASY",
        "PCD_RAS_MED"]


def test_nearest_neighbors_different_frame(warm_up, lower_bound_idx,distance_threshold, top_k):
    """Test nearest neighbor computation with different frame IDs."""
    
    for seq in SEQs:
        seq_dir = os.path.join(ROOT_DIR, seq)
        print(f"\n{'='*80}")
        print(f"Testing nearest neighbors for: {seq}")
        print(f"{'='*80}")
        
        fs = file_structure(ROOT_DIR, seq, verbose=True)
        
        # Get positions, labels, and frame IDs
        positions = fs._get_positions_()
        labels = fs._get_labels()
        frame_ids = fs._get_frame_ids()
        
        print(f"\nDataset Statistics:")
        print(f"  Total positions: {len(positions)}")
        print(f"  Unique frame IDs: {np.unique(frame_ids)}")
        print(f"  Unique labels: {np.unique(labels)}")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
        
        # Get improved nearest neighbor ground truth
        print(f"\nComputing nearest neighbors (ground truth loop closures)...")
        gt_nn = fs.get_ground_truth_loop_closure(
            warm_up=warm_up,
            lower_bound_idx=lower_bound_idx,
            distance_threshold=distance_threshold,
            topk=top_k  # Use all neighbors within threshold
        )

        print(f"\nGround Truth Loop Closure Summary:")
        print(f"  Total query positions: {gt_nn['total_query_positions']}")
        print(f"  Valid loop closures: {gt_nn['valid_loop_closures']}")
        print(f"  Query indices range: [{min(gt_nn['query_indices'])}, {max(gt_nn['query_indices'])}]")
        print(f"  Neighbor indices range: [{min(gt_nn['neighbor_indices'])}, {max(gt_nn['neighbor_indices'])}]")
        
        print(f"\nDistance Statistics:")
        stats = gt_nn['statistics']
        print(f"  Min distance: {stats['min_distance']:.4f}m")
        print(f"  Max distance: {stats['max_distance']:.4f}m")
        print(f"  Mean distance: {stats['mean_distance']:.4f}m")
        print(f"  Median distance: {stats['median_distance']:.4f}m")
        
        # Show first 10 loop closures
        print(f"\nFirst 10 loop closure pairs:")
        print(f"{'Query Index':<15} {'Neighbor Index':<15} {'Distance (m)':<15} {'Label':<10}")
        print(f"{'-'*55}")
        for i in range(min(10, len(gt_nn['query_indices']))):
            query_idx = int(gt_nn['query_indices'][i])
            neighbor_idx = int(gt_nn['neighbor_indices'][i])
            distance = float(gt_nn['distances'][i])
            label = int(gt_nn['labels'][i])
            print(f"{query_idx:<15} {neighbor_idx:<15} {distance:<15.4f} {label:<10}")
        
        # Get detailed nearest neighbors
        #all_nn = fs.compute_all_nearest_neighbors_different_frame(lower_bound_idx=50)
        
        print(f"\nGround Truth Loop Closure Analysis:")
        print(f"  Total unique query positions: {gt_nn['total_query_positions']}")
        print(f"  Total loop closures found: {gt_nn['valid_loop_closures']}")
        
        # Save ground truth loop closures
        print(f"\nSaving ground truth loop closures...")
        saved_files = fs.save_ground_truth_loop_closures(
            output_dir=os.path.join(ROOT_DIR, seq, 'ground_truth'),
            warm_up=warm_up,
            lower_bound_idx=lower_bound_idx,
            distance_threshold=distance_threshold,
            topk=top_k # Save all neighbors within threshold
        )
        
        # Plot 3D path with nearest neighbor connections
        plot_nearest_neighbors_3d(positions, gt_nn, seq, samples=10)


def plot_nearest_neighbors_3d(positions, all_neighbors_dict, seq_name, samples=10):
    """
    Plot 3D path with connections to nearest neighbors of different frame IDs.
    
    Args:
        positions: Nx3 array of positions
        all_neighbors_dict: Dictionary from compute_all_nearest_neighbors_different_frame
        seq_name: Sequence name for title
        samples: Number of connections to sample for visualization
    """
    # Align and elevate positions
    aligned_positions = aligned_path(positions)
    elevated_positions = elevate_along_path(aligned_positions, max_elevation=20.0)
    
    # Extract data from dictionary
    query_indices = all_neighbors_dict['query_indices']
    neighbor_indices = all_neighbors_dict['neighbor_indices']
    distances = all_neighbors_dict['distances']
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot path line
    ax.plot(elevated_positions[:, 0], elevated_positions[:, 1], elevated_positions[:, 2],
            'k-', alpha=0.6, linewidth=1.5, label='Path trajectory')
    
    # Plot all positions as small dots
    ax.scatter(elevated_positions[:, 0], elevated_positions[:, 1], elevated_positions[:, 2],
              c='lightblue', s=5, alpha=0.4)
    
    # Plot connections to nearest neighbors (sample for clarity)
    sample_rate = max(1, len(query_indices) // samples)
    connection_count = 0
    
    for idx in range(0, len(query_indices), sample_rate):
        query_idx = int(query_indices[idx])
        neighbor_idx = neighbor_indices[idx]
        distance = distances[idx]
        
        # Skip if no valid neighbor
        if neighbor_idx is None or distance == np.inf:
            continue
        
        neighbor_idx = int(neighbor_idx)
        query_pos = elevated_positions[query_idx]
        nn_pos = elevated_positions[neighbor_idx]
        
        # Draw line from query point to nearest neighbor
        ax.plot([query_pos[0], nn_pos[0]], 
               [query_pos[1], nn_pos[1]], 
               [query_pos[2], nn_pos[2]],
               'g-', alpha=0.7, linewidth=2.0)
        
        # Mark the connection endpoints
        ax.scatter(*query_pos, color='red', s=100, alpha=0.7, marker='o', 
                  edgecolors='darkred', linewidth=1.5, label='Query' if connection_count == 0 else '')
        ax.scatter(*nn_pos, color='orange', s=100, alpha=0.7, marker='s', 
                  edgecolors='darkorange', linewidth=1.5, label='Neighbor' if connection_count == 0 else '')
        
        connection_count += 1
    
    # Set labels and title
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)
    ax.set_zlabel("Z (Elevation, m)", fontsize=12)
    ax.set_title(f"Ground Truth Loop Closures - {seq_name}\n({connection_count} connections shown)", 
                fontsize=14, fontweight='bold')
    
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
    
    # Add legend and grid
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print("Testing nearest neighbor computation with different frame IDs...")
    test_nearest_neighbors_different_frame(warm_up=100, lower_bound_idx=50, distance_threshold=10, top_k=1)
    print("\nAll tests completed!")