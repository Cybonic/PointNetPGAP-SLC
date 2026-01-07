import argparse

def create_argument_parser():
    """
    Create and return an argument parser with all parameters for gen_descriptors.py
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate descriptors for place recognition using PointNetGAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gen_descriptors.py --dataset_root /path/to/dataset --network SPVSoAP3D
  python gen_descriptors.py --dataset_root /path/to/dataset --session hortov2 --device cuda:0
  python gen_descriptors.py --dataset_root /path/to/dataset --batch_size 32 --max_points 5000
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=True,
        help='Directory to the dataset root'
    )
    
    # Dataset parameters
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='HORTOv2',
        help='Dataset name (default: HORTOv2)'
    )
    
    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default=None,
        help='Validation set name'
    )
    
    # Network and model parameters
    parser.add_argument(
        '--network',
        type=str,
        default='SPVSoAP3D',
        help='Model to be used (default: SPVSoAP3D)'
    )
    
    parser.add_argument(
        '--resume',
        '-r',
        type=str,
        required=False,
        default='checkpoints/SPCoV/predictions',
        help='Directory to get the trained model or descriptors'
    )
    
    # Session and experiment configuration
    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default='ukfrpt',
        help='Session configuration name (YAML file in sessions/ directory)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        default='uk',
        help='Name of the experiment to be executed (default: uk)'
    )
    
    # Hardware and computation parameters
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (default: cuda). Can specify cuda:0, cuda:1, cpu, etc.'
    )
    
    parser.add_argument(
        '--memory',
        type=str,
        default='DISK',
        choices=['DISK', 'RAM'],
        help='RAM: loads dataset to RAM. DISK: loads on the fly from disk (default: DISK)'
    )
    
    # Data processing parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for inference (default: 10)'
    )
    
    parser.add_argument(
        '--max_points',
        type=int,
        default=10000,
        help='Maximum number of points to sample from point clouds (default: 10000)'
    )
    
    parser.add_argument(
        '--roi',
        type=float,
        required=False,
        default=0,
        help='Crop range [m] to crop the point cloud around the scan origin (default: 0, no crop)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--eval_file',
        type=str,
        required=False,
        default='eval/ground_truth_loop_range_10m.pkl',
        help='Path to ground truth evaluation file (default: eval/ground_truth_loop_range_10m.pkl)'
    )
    
    parser.add_argument(
        '--monitor_loop_range',
        type=float,
        required=False,
        default=10,
        help='Loop range [m] to monitor performance (default: 10)'
    )
    
    parser.add_argument(
        '--eval_roi_window',
        type=float,
        required=False,
        default=100,
        help='Number of frames to ignore in immediate vicinity of query frame (default: 100)'
    )
    
    parser.add_argument(
        '--eval_warmup_window',
        type=float,
        required=False,
        default=100,
        help='Number of frames to ignore at the beginning of the sequence (default: 100)'
    )
    
    parser.add_argument(
        '--eval_protocol',
        type=str,
        required=False,
        choices=['place'],
        default='place',
        help='Evaluation protocol to use (default: place)'
    )
    
    # Output parameters
    parser.add_argument(
        '--save_predictions',
        type=str,
        required=False,
        default='saved_model_data',
        help='Directory to save generated descriptors (default: saved_model_data)'
    )
    
    # Descriptor parameters
    parser.add_argument(
        '--descriptor_size',
        type=int,
        required=False,
        default=256,
        help='Size of the descriptor vector (default: 256)'
    )
    
    return parser


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
        list: Unparsed arguments
    """
    parser = create_argument_parser()
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


def print_arguments(FLAGS):
    """
    Print parsed arguments in a formatted manner.
    
    Args:
        FLAGS: Parsed arguments
    """
    print("\n" + "="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    
    print("\n--- Dataset Configuration ---")
    print(f"Dataset Root: {FLAGS.dataset_root}")
    print(f"Dataset: {FLAGS.dataset}")
    print(f"Validation Set: {FLAGS.val_set}")
    
    print("\n--- Model Configuration ---")
    print(f"Network: {FLAGS.network}")
    print(f"Resume From: {FLAGS.resume}")
    print(f"Session: {FLAGS.session}")
    print(f"Experiment: {FLAGS.experiment}")
    
    print("\n--- Hardware Configuration ---")
    print(f"Device: {FLAGS.device}")
    print(f"Memory Mode: {FLAGS.memory}")
    
    print("\n--- Data Processing ---")
    print(f"Batch Size: {FLAGS.batch_size}")
    print(f"Max Points: {FLAGS.max_points}")
    print(f"ROI Crop Range: {FLAGS.roi}m")
    print(f"Descriptor Size: {FLAGS.descriptor_size}")
    
    print("\n--- Evaluation Configuration ---")
    print(f"Eval File: {FLAGS.eval_file}")
    print(f"Monitor Loop Range: {FLAGS.monitor_loop_range}m")
    print(f"ROI Window: {FLAGS.eval_roi_window} frames")
    print(f"Warmup Window: {FLAGS.eval_warmup_window} frames")
    print(f"Eval Protocol: {FLAGS.eval_protocol}")
    
    print("\n--- Output Configuration ---")
    print(f"Save Predictions To: {FLAGS.save_predictions}")
    
    print("="*60 + "\n")


# Example usage in main script
if __name__ == '__main__':
    # Parse arguments
    FLAGS, unparsed = parse_arguments()
    
    # Print arguments
    print_arguments(FLAGS)
    
    # Print any unparsed arguments
    if unparsed:
        print(f"Unparsed arguments: {unparsed}")