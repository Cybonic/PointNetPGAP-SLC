#!/usr/bin/env python3
"""
Batch evaluation script for ScanContext on HortoV2 dataset sequences.

Similar to script_eval_hortov2.py but for ScanContext descriptors.

Author: GitHub Copilot
Date: 2026-01-20
"""

import os
import sys
import subprocess

# Define test sequences
test_sequences = [
    'PCD_Easy_DARK',
    'PCD_MED',
    #'PCD_EASY',
    #'PCD_RAS_EASY'
]

# Get root directory (PointNetGAP folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(script_dir, '..'))
print("Root directory:", root)
print("Script directory:", script_dir)

# Path to the scancontext script (it's in the same directory as this script)
scancontext_script = os.path.join(script_dir, 'scancontext_hortov2.py')

# Dataset root path
dataset_root = os.path.join(root, 'dataset', 'PlaceRecognitionTestPolyTunnel')

# Different ScanContext configurations to test
sc_configs = [
    {'sector_res': 60, 'ring_res': 20, 'max_length': 80},   # Default
    {'sector_res': 90, 'ring_res': 20, 'max_length': 80},   # More sectors
    {'sector_res': 60, 'ring_res': 40, 'max_length': 80},   # More rings
]

def run_scancontext_eval(sequence, config):
    """
    Run ScanContext evaluation for a given sequence and configuration.
    
    Args:
        sequence: Sequence name
        config: Dictionary with ScanContext parameters
    """
    print("\n" + "="*80)
    print(f"Processing sequence: {sequence}")
    print(f"Configuration: sector_res={config['sector_res']}, ring_res={config['ring_res']}")
    print("="*80 + "\n")
    
    # Build command
    cmd = [
        'python', scancontext_script,
        '--dataset_root', dataset_root,
        '--sequence', sequence,
        '--sector_res', str(config['sector_res']),
        '--ring_res', str(config['ring_res']),
        '--max_length', str(config['max_length']),
        '--top_k', '25',
        '--output_dir', 'saved/hortov2'  # Save to PointNetGAP's saved folder
    ]
    
    # Execute command
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✓ Successfully processed {sequence}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error processing {sequence}: {e}")
        return False
    
    return True


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("ScanContext Batch Evaluation for HortoV2 Dataset")
    print("="*80 + "\n")
    
    # Check if script exists
    if not os.path.exists(scancontext_script):
        print(f"Error: ScanContext script not found at {scancontext_script}")
        return
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"Warning: Dataset root not found at {dataset_root}")
        print("Continuing anyway (may fail if path is incorrect)")
    
    # Process each sequence
    results = {}
    
    # Use only the default configuration for now
    config = sc_configs[0]
    
    for seq in test_sequences:
        success = run_scancontext_eval(seq, config)
        results[seq] = success
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for seq, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{seq:20s} : {status}")
    
    total = len(results)
    successful = sum(results.values())
    print(f"\nTotal: {successful}/{total} sequences processed successfully")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
