"""
Comprehensive test demonstrating ground truth loop closure generation with different modes.
Tests both distance threshold and top-K selection methods.
"""

import sys
import os
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure

ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"
SEQUENCE = "PCD_EASY"


def test_distance_threshold_mode():
    """Test ground truth generation with distance threshold mode."""
    print("\n" + "="*90)
    print("MODE 1: DISTANCE THRESHOLD")
    print("="*90)
    
    fs = file_structure(ROOT_DIR, SEQUENCE, verbose=False)
    
    # Generate ground truth with distance threshold
    gt = fs.get_ground_truth_loop_closure(
        warm_up=50,
        lower_bound_idx=20,
        distance_threshold=2.0,  # All neighbors within 2 meters
        topk=None
    )
    
    print(f"\nConfiguration:")
    print(f"  warm_up: 50 (skip first 50 frames)")
    print(f"  lower_bound_idx: 20 (ignore last 20 frames for same trajectory)")
    print(f"  distance_threshold: 2.0 m")
    print(f"  topk: None (all within threshold)")
    
    print(f"\nResults:")
    print(f"  Total query positions: {gt['total_query_positions']}")
    print(f"  Total loop closures: {gt['valid_loop_closures']}")
    print(f"  Avg neighbors per query: {gt['valid_loop_closures'] / max(1, gt['total_query_positions']):.2f}")
    
    print(f"\nDistance Statistics:")
    stats = gt['statistics']
    print(f"  Min: {stats['min_distance']:.4f} m")
    print(f"  Max: {stats['max_distance']:.4f} m")
    print(f"  Mean: {stats['mean_distance']:.4f} m")
    print(f"  Median: {stats['median_distance']:.4f} m")
    print(f"  Std: {stats['std_distance']:.4f} m")
    
    # Show sample pairs
    print(f"\nFirst 5 query positions with their neighbors:")
    for query_idx in list(gt['query_to_neighbor'].keys())[:5]:
        neighbors = gt['query_to_neighbor'][query_idx]
        print(f"  Query {query_idx}: {len(neighbors)} neighbors")
        for n in neighbors[:3]:  # Show first 3
            print(f"    -> Neighbor {n['neighbor_idx']}: {n['distance']:.4f}m")
        if len(neighbors) > 3:
            print(f"    ... and {len(neighbors) - 3} more")
    
    return gt


def test_topk_mode():
    """Test ground truth generation with top-K mode."""
    print("\n" + "="*90)
    print("MODE 2: TOP-K SELECTION")
    print("="*90)
    
    fs = file_structure(ROOT_DIR, SEQUENCE, verbose=False)
    
    # Generate ground truth with top-K
    for topk_value in [1, 5, 10]:
        gt = fs.get_ground_truth_loop_closure(
            warm_up=50,
            lower_bound_idx=20,
            distance_threshold=10.0,  # Upper bound (all eligible)
            topk=topk_value
        )
        
        print(f"\nTop-K = {topk_value}:")
        print(f"  Total loop closures: {gt['valid_loop_closures']}")
        print(f"  Unique queries: {gt['total_query_positions']}")
        print(f"  Avg neighbors per query: {gt['valid_loop_closures'] / max(1, gt['total_query_positions']):.2f}")
        print(f"  Distance stats:")
        stats = gt['statistics']
        print(f"    Mean: {stats['mean_distance']:.4f} m")
        print(f"    Max: {stats['max_distance']:.4f} m")


def test_parameter_combinations():
    """Test various parameter combinations and show their effects."""
    print("\n" + "="*90)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*90)
    
    fs = file_structure(ROOT_DIR, SEQUENCE, verbose=False)
    
    test_cases = [
        {"name": "Conservative", "warm_up": 100, "lower_bound_idx": 50, "dist": 1.0, "topk": None},
        {"name": "Moderate", "warm_up": 50, "lower_bound_idx": 20, "dist": 2.0, "topk": None},
        {"name": "Lenient", "warm_up": 30, "lower_bound_idx": 10, "dist": 5.0, "topk": None},
        {"name": "Very Lenient", "warm_up": 20, "lower_bound_idx": 5, "dist": 10.0, "topk": None},
        {"name": "Top-5 Mode", "warm_up": 50, "lower_bound_idx": 20, "dist": 10.0, "topk": 5},
        {"name": "Top-10 Mode", "warm_up": 50, "lower_bound_idx": 20, "dist": 10.0, "topk": 10},
    ]
    
    print(f"\n{'Configuration':<20} {'Queries':<10} {'Closures':<12} {'Avg/Q':<10} {'Mean(m)':<10}")
    print("-" * 70)
    
    for case in test_cases:
        gt = fs.get_ground_truth_loop_closure(
            warm_up=case['warm_up'],
            lower_bound_idx=case['lower_bound_idx'],
            distance_threshold=case['dist'],
            topk=case['topk']
        )
        
        avg_per_query = gt['valid_loop_closures'] / max(1, gt['total_query_positions'])
        mean_dist = gt['statistics']['mean_distance']
        
        topk_str = f"Top-{case['topk']}" if case['topk'] else f"Thresh:{case['dist']}"
        print(f"{case['name']:<20} {gt['total_query_positions']:<10} {gt['valid_loop_closures']:<12} "
              f"{avg_per_query:<10.2f} {mean_dist:<10.4f}")


def test_file_saving():
    """Test saving ground truth with both modes."""
    print("\n" + "="*90)
    print("FILE SAVING TEST")
    print("="*90)
    
    fs = file_structure(ROOT_DIR, SEQUENCE, verbose=False)
    
    # Save with threshold mode
    print(f"\nSaving with DISTANCE THRESHOLD mode...")
    output_dir_threshold = os.path.join(ROOT_DIR, SEQUENCE, "gt_test_threshold")
    files_threshold = fs.save_ground_truth_loop_closures(
        output_dir=output_dir_threshold,
        warm_up=100,
        lower_bound_idx=50,
        distance_threshold=2.0,
        topk=1
    )
    
    # Save with topk mode
    print(f"\nSaving with TOP-K mode...")
    output_dir_topk = os.path.join(ROOT_DIR, SEQUENCE, "gt_test_topk")
    files_topk = fs.save_ground_truth_loop_closures(
        output_dir=output_dir_topk,
        warm_up=50,
        lower_bound_idx=20,
        distance_threshold=10.0,
        topk=5
    )
    
    # Compare file sizes
    print(f"\n{'File Type':<30} {'Threshold Mode (KB)':<20} {'Top-K Mode (KB)':<20}")
    print("-" * 70)
    
    for file_type in ['csv', 'npz', 'json', 'stats', 'txt']:
        if file_type in files_threshold and file_type in files_topk:
            size_threshold = os.path.getsize(files_threshold[file_type]) / 1024
            size_topk = os.path.getsize(files_topk[file_type]) / 1024
            print(f"{file_type:<30} {size_threshold:<20.1f} {size_topk:<20.1f}")
    
    # Load and display parameters from saved files
    print(f"\nSaved Parameters (Threshold Mode):")
    with open(files_threshold['parameters'], 'r') as f:
        params = json.load(f)
        for key, value in params['computation_parameters'].items():
            print(f"  {key}: {value}")


def main():
    """Run all tests."""
    print("\n" + "█"*90)
    print("GROUND TRUTH LOOP CLOSURE GENERATION - COMPREHENSIVE TEST")
    print("█"*90)
    
    # Test both modes
    gt_threshold = test_distance_threshold_mode()
    test_topk_mode()
    
    # Parameter sensitivity analysis
    test_parameter_combinations()
    
    # File saving
    test_file_saving()
    
    print("\n" + "█"*90)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("█"*90 + "\n")


if __name__ == '__main__':
    main()
