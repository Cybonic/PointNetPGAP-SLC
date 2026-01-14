#!/usr/bin/env python3
"""
Quick example: Generate a single loop closure GIF using predictions.

This is a simplified example for testing the visualization with specific parameters.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from PointNetGAP.dataloader.hortov2.dataset import file_structure
import pickle


def load_predictions(prediction_file: str):
    """Load predictions from a pickle file."""
    if not os.path.exists(prediction_file):
        raise FileNotFoundError(f"Prediction file not found: {prediction_file}")
    
    with open(prediction_file, 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded predictions from: {prediction_file}")
    print(f"  Number of queries: {len(predictions)}")
    
    return predictions


# Import the visualizer class
from plot_loop_closure_pr_gif import LoopClosureVisualizerPredictions


def generate_single_gif_example():
    """Generate a single GIF for demonstration."""
    
    # Configuration
    DATASET_ROOT_DIR = "/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel"
    SEQUENCE = "PCD_Easy_DARK"
    MODEL = "PointNetPGAP-None"
    
    # Prediction file path
    pred_file = f"/home/tiago/workspace/place_uk/PointNetGAP/saved/hortov2/{SEQUENCE}/{MODEL}/predictions/place/0.636@1/predictions.pkl"
    
    # Visualization parameters
    TOPK = 3  # Show top-3 predictions
    DISTANCE_THRESHOLD = 15.0  # 15 meters
    SAMPLE_RATE = 5  # Show every 5th query
    
    print("=" * 80)
    print("Single GIF Example - Loop Closure Visualization")
    print("=" * 80)
    print(f"Sequence: {SEQUENCE}")
    print(f"Model: {MODEL}")
    print(f"Top-K: {TOPK}")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}m")
    print(f"Sample rate: {SAMPLE_RATE}")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading dataset...")
    fs = file_structure(DATASET_ROOT_DIR, SEQUENCE, verbose=False)
    print(f"  Positions: {len(fs._get_positions_())}")
    
    # Load predictions
    print("\nLoading predictions...")
    predictions = load_predictions(pred_file)
    
    # Create visualizer
    print("\nCreating visualizer...")
    visualizer = LoopClosureVisualizerPredictions(
        fs, 
        predictions, 
        f"{SEQUENCE}_{MODEL}_example",
        topk=TOPK,
        distance_threshold=DISTANCE_THRESHOLD,
        sample_rate=SAMPLE_RATE
    )
    
    # Generate GIF
    output_path = f"example_loop_closure_pred_{SEQUENCE}_topk{TOPK}.gif"
    print(f"\nGenerating GIF: {output_path}")
    visualizer.generate_gif(output_path, fps=3, dpi=100)
    
    print("\n" + "=" * 80)
    print("✓ Example GIF generated successfully!")
    print(f"  Output: {os.path.abspath(output_path)}")
    print("=" * 80)


if __name__ == '__main__':
    generate_single_gif_example()
