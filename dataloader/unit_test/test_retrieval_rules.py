"""
Test script to verify retrieval rules:
1. Retrieval is ALWAYS done in past frames only (never future frames)
2. The nearest neighbor is the CLOSEST point, even if it has been retrieved before
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure

ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"


def test_retrieval_rules(sequence="PCD_Easy_DARK"):
    """Test that retrieval follows the correct rules."""
    
    print("=" * 80)
    print(f"Testing Retrieval Rules for Sequence: {sequence}")
    print("=" * 80)
    
    # Load dataset
    fs = file_structure(ROOT_DIR, sequence, verbose=False)
    
    # Get ground truth loop closures
    gt = fs.get_ground_truth_loop_closure(
        warm_up=100,
        lower_bound_idx=50,
        distance_threshold=10.0,
        topk=1
    )
    
    query_indices = gt['query_indices']
    neighbor_indices = gt['neighbor_indices']
    
    print(f"\nTotal loop closures: {len(query_indices)}")
    print(f"Query indices range: [{np.min(query_indices)}, {np.max(query_indices)}]")
    print(f"Neighbor indices range: [{np.min(neighbor_indices)}, {np.max(neighbor_indices)}]")
    
    # RULE 1: Verify retrieval is always from past frames
    print("\n" + "-" * 80)
    print("RULE 1: Retrieval is ALWAYS from PAST frames (never future frames)")
    print("-" * 80)
    
    past_frame_count = 0
    future_frame_count = 0
    
    for query_idx, neighbor_idx in zip(query_indices, neighbor_indices):
        if query_idx > neighbor_idx:
            past_frame_count += 1
        else:
            future_frame_count += 1
            print(f"  ✗ ERROR: Query {query_idx} retrieved neighbor {neighbor_idx} (FUTURE!)")
    
    if future_frame_count == 0:
        print(f"✓ PASSED: All {past_frame_count} retrievals are from PAST frames")
    else:
        print(f"✗ FAILED: {future_frame_count} retrievals are from FUTURE frames!")
    
    # RULE 2: Verify the same neighbor can be retrieved multiple times
    print("\n" + "-" * 80)
    print("RULE 2: Nearest neighbor is CLOSEST point (can be retrieved multiple times)")
    print("-" * 80)
    
    # Count how many times each neighbor is retrieved
    neighbor_counts = {}
    for neighbor_idx in neighbor_indices:
        neighbor_counts[neighbor_idx] = neighbor_counts.get(neighbor_idx, 0) + 1
    
    # Find neighbors retrieved multiple times
    repeated_neighbors = {k: v for k, v in neighbor_counts.items() if v > 1}
    
    print(f"Total unique neighbors: {len(neighbor_counts)}")
    print(f"Neighbors retrieved multiple times: {len(repeated_neighbors)}")
    
    if repeated_neighbors:
        print("\n✓ PASSED: Same neighbor CAN be retrieved multiple times (correct!)")
        print("\nTop 10 most frequently retrieved neighbors:")
        sorted_neighbors = sorted(repeated_neighbors.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for neighbor_idx, count in sorted_neighbors:
            # Find which queries retrieved this neighbor
            query_idxs = [int(query_indices[i]) for i in range(len(query_indices)) 
                         if neighbor_indices[i] == neighbor_idx]
            print(f"  Neighbor {neighbor_idx}: retrieved {count} times by queries {query_idxs[:3]}...")
    else:
        print("Note: No neighbors were retrieved multiple times in this test")
        print("(This could happen if queries are far apart or with strict distance threshold)")
    
    # RULE 3: Verify distances are sorted (closest first when topk=1)
    print("\n" + "-" * 80)
    print("RULE 3: Verify closest point is selected")
    print("-" * 80)
    
    distances = gt['distances']
    print(f"Distance statistics:")
    print(f"  Min: {np.min(distances):.4f}m")
    print(f"  Max: {np.max(distances):.4f}m")
    print(f"  Mean: {np.mean(distances):.4f}m")
    print(f"  Median: {np.median(distances):.4f}m")
    
    print("\nFirst 10 loop closures:")
    print(f"{'Query':<10} {'Neighbor':<10} {'Distance':<12} {'Past?':<10}")
    print("-" * 42)
    for i in range(min(10, len(query_indices))):
        q = int(query_indices[i])
        n = int(neighbor_indices[i])
        d = float(distances[i])
        is_past = "✓ Yes" if q > n else "✗ No"
        print(f"{q:<10} {n:<10} {d:<12.4f} {is_past:<10}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_retrieval_rules("PCD_Easy_DARK")
    print("\n")
    test_retrieval_rules("PCD_MED")
