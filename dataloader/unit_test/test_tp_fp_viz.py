#!/usr/bin/env python3
"""
Test the TP/FP visualization with a single frame.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import the updated visualizer
exec(open('/home/tiago/workspace/place_uk/PointNetGAP/dataloader/unit_test/plot_loop_closure_pr_gif.py').read().split('if __name__')[0])

# Configuration
DATASET_ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"
SEQUENCE = "PCD_Easy_DARK"
MODEL = "PointNetPGAP-None"
pred_file = f"/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2/{SEQUENCE}/{MODEL}/predictions/place/0.636@1/predictions.pkl"

print("Loading dataset and predictions...")
fs = file_structure(DATASET_ROOT_DIR, SEQUENCE, verbose=False)

with open(pred_file, 'rb') as f:
    predictions = pickle.load(f)

print(f"Loaded {len(predictions)} predictions")

# Create visualizer with top-5 to see both TP and FP
visualizer = LoopClosureVisualizerPredictions(
    fs, predictions, f"{SEQUENCE}_{MODEL}_test",
    topk=5,
    distance_threshold=10.0,
    sample_rate=1
)

# Find a query with both TP and FP
test_query_idx = None
for query_idx in list(predictions.keys())[:100]:
    frame_data = visualizer.get_frame_data(query_idx)
    if frame_data['num_true_positives'] > 0 and frame_data['num_false_positives'] > 0:
        test_query_idx = query_idx
        break

if test_query_idx:
    print(f"\nTesting with query frame: {test_query_idx}")
    frame_data = visualizer.get_frame_data(test_query_idx)
    print(f"  Query segment: {frame_data['query_segment']}")
    print(f"  True Positives: {frame_data['num_true_positives']}")
    print(f"  False Positives: {frame_data['num_false_positives']}")
    print(f"  Neighbor segments: {frame_data['neighbor_segments']}")
    print(f"  Is TP: {frame_data['is_true_positive']}")
    
    # Create single frame visualization
    fig, ax = visualizer.create_figure(figsize=(14, 10))
    visualizer.plot_frame(ax, test_query_idx, frame_data, show_connections=True, show_trajectory=True)
    
    output_path = "test_tp_fp_visualization.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Test visualization saved: {output_path}")
    print("  Green circles with 'TP' = True Positives (correct segment)")
    print("  Red circles with 'FP' = False Positives (wrong segment)")
else:
    print("No query found with both TP and FP in first 100 frames")
