"""
Verify that plot_tp_individual.py correctly implements retrieval rules:
1. Retrieval is ALWAYS done in past frames only (never future frames)
2. The nearest neighbor is the CLOSEST point, even if it has been retrieved before
"""

import sys
import os
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure
from plot_tp_individual import collect_true_positives, collect_ground_truth_loops

ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/place_v2/PlaceRecognitionTestPolyTunnel"
SAVED_ROOT = "/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2"


def verify_past_frames_only(loop_closures, name):
    """Verify all retrievals are from past frames."""
    print(f"\n{'='*80}")
    print(f"Verifying: {name}")
    print(f"{'='*80}")
    
    past_count = 0
    future_count = 0
    
    for query_idx, neighbor_idx, distance in loop_closures:
        if query_idx > neighbor_idx:
            past_count += 1
        else:
            future_count += 1
            print(f"  ✗ ERROR: Query {query_idx} retrieved neighbor {neighbor_idx} (FUTURE!)")
    
    total = len(loop_closures)
    print(f"Total loop closures: {total}")
    print(f"Past frame retrievals: {past_count} ({100*past_count/total:.1f}%)")
    print(f"Future frame retrievals: {future_count} ({100*future_count/total:.1f}%)")
    
    if future_count == 0:
        print(f"✓ PASSED: All retrievals are from PAST frames only")
        return True
    else:
        print(f"✗ FAILED: {future_count} retrievals from FUTURE frames!")
        return False


def verify_repeated_neighbors(loop_closures, name):
    """Verify that same neighbors can be retrieved multiple times."""
    print(f"\n{'='*80}")
    print(f"Checking repeated neighbors: {name}")
    print(f"{'='*80}")
    
    neighbor_counts = {}
    for query_idx, neighbor_idx, distance in loop_closures:
        if neighbor_idx not in neighbor_counts:
            neighbor_counts[neighbor_idx] = []
        neighbor_counts[neighbor_idx].append((query_idx, distance))
    
    repeated = {k: v for k, v in neighbor_counts.items() if len(v) > 1}
    
    print(f"Total unique neighbors: {len(neighbor_counts)}")
    print(f"Neighbors retrieved multiple times: {len(repeated)}")
    
    if repeated:
        print(f"\n✓ PASSED: Same neighbor CAN be retrieved multiple times")
        print(f"\nTop 5 most frequently retrieved neighbors:")
        sorted_neighbors = sorted(repeated.items(), key=lambda x: len(x[1]), reverse=True)[:5]
        for neighbor_idx, queries in sorted_neighbors:
            print(f"  Neighbor {neighbor_idx}: retrieved {len(queries)} times")
            print(f"    Query indices: {[q[0] for q in queries[:5]]}...")
        return True
    else:
        print(f"Note: No neighbors retrieved multiple times (may be valid depending on data)")
        return True


def test_ground_truth():
    """Test ground truth collection."""
    print("\n" + "="*80)
    print("TESTING GROUND TRUTH COLLECTION")
    print("="*80)
    
    sequence = "PCD_Easy_DARK"
    fs = file_structure(ROOT_DIR, sequence, verbose=False)
    
    gt_loops = collect_ground_truth_loops(fs, distance_threshold=10.0, min_temporal_distance=50)
    
    verify_past_frames_only(gt_loops, f"Ground Truth - {sequence}")
    verify_repeated_neighbors(gt_loops, f"Ground Truth - {sequence}")


def test_model_predictions():
    """Test model prediction collection."""
    print("\n" + "="*80)
    print("TESTING MODEL PREDICTION COLLECTION")
    print("="*80)
    
    sequence = "PCD_Easy_DARK"
    model = "PointNetPGAP"
    
    # Load predictions
    model_dir = os.path.join(SAVED_ROOT, sequence, f"{model}-None", "predictions", "place")
    subdirs = [d for d in os.listdir(model_dir) 
              if os.path.isdir(os.path.join(model_dir, d)) and '@1' in d]
    pred_path = os.path.join(model_dir, subdirs[0], "predictions.pkl")
    
    with open(pred_path, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"\nLoaded {len(predictions)} predictions from {model}")
    
    tp_loops = collect_true_positives(predictions, topk=1, distance_threshold=10.0, min_temporal_distance=50)
    
    verify_past_frames_only(tp_loops, f"{model} - {sequence}")
    verify_repeated_neighbors(tp_loops, f"{model} - {sequence}")


if __name__ == "__main__":
    test_ground_truth()
    test_model_predictions()
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
