#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/


'''

Version: 3.1 
 - pretrained model is automatically loaded based on the model and session names 
 
'''
import os, json, math, logging, datetime
import argparse
import yaml
import os
import torch 

from trainer import Trainer
from pipeline_factory import model_handler,dataloader_handler
import numpy as np
from place_recognition import PlaceRecognition
from utils import logger


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

torch.cuda.empty_cache()
torch.autograd.set_detect_anomaly(True)
# On terminal run the following command to set the environment variable
# export CUBLAS_WORKSPACE_CONFIG=":4096:8"

#torch.use_deterministic_algorithms(True)
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
        default='dataset/PlaceRecognitionTestPolyTunnel',
        help='Directory to the dataset root'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        default='hortov2',
        help='Experiment name'
    )
    
    parser.add_argument(
        '--session',
        type=str,
        default='hortov2',
        help='Session name for the experiment'
    )
    
    parser.add_argument(
        '--network',
        type=str,
        default='PointNetPGAP',
        help='Network architecture to use'
    )

    return parser.parse_known_args()

def plot_session(SESSION):
    """
    Display the session configuration in a formatted table.
    
    Args:
        SESSION: Dictionary containing session configuration
    """
    print("\n")
    print("=" * 80)
    print("SESSION CONFIGURATION")
    print("=" * 80)
    
    def print_dict(d, indent=0):
        """Recursively print nested dictionaries."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                print_dict(value, indent + 1)
            elif isinstance(value, list):
                print(f"{'  ' * indent}{key}: {value}")
            else:
                print(f"{'  ' * indent}{key}: {value}")
    
    print_dict(SESSION)
    
    print("=" * 80)
    print("\n")

if __name__ == '__main__':

    FLAGS, unparsed = create_argument_parser()

    # load config file at session folder
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # get parent dir 
    print("Root directory:", root)
    session_cfg_file = os.path.join(root,'PointNetGAP','sessions', FLAGS.session + '.yaml')
    assert os.path.exists(session_cfg_file), "Session config file not found"
    # global path
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    # plot_session(SESSION)
    
    device = SESSION['device']
    # For repeatability
    
    torch.manual_seed(0)
    np.random.seed(0)

    ######################################################################

    # Build the model and the loader
    model = model_handler(  network = SESSION['network'],
                            device  = device,
                            )


    loader = dataloader_handler(root, 
                                network = SESSION['network'],
                                val_loader = SESSION['val_loader'],
                                train_loader = SESSION['train_loader'],
                                eval_protocol= SESSION['run']['eval_protocol'],
                                )

    run_name = {'experiment': str(FLAGS.experiment), 
                'seq':SESSION['val_loader']['dataset']['seq'][0],
                'model': str(model)
            }

    # run_name_str = '_'.join([f"{v}" for k, v in run_name.items()])
    os.makedirs('logs', exist_ok=True)
    experiment_name_log =  '-'.join([f"{v}" for k, v in run_name.items()])

    log_file = os.path.join('logs',f'{experiment_name_log}.log')
    logger = logging.getLogger(__name__)
    log_handler = logging.FileHandler(log_file)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    

    retrieval  = SESSION['retrieval']
    loader_val = loader.get_val_loader()
    run_config = SESSION['run']

    eval_approach = PlaceRecognition(model ,loader_val, retrieval,
                                    logger, run_config, run_name,
                                    device)
    
    # Define a set of loop ranges to be evaluated
    loop_range = list(range(0,120,1))
    
    warm_up = retrieval['warmup_window']
    topk = retrieval['top_cand']
    roi_window = retrieval['roi_window']
    sim_metric = retrieval['sim_metric']
    

    #assert os.path.exists(FLAGS.resume ), "File not found %s"%FLAGS.resume 
    resume = os.path.join(root,'PointNetGAP',SESSION['network']['checkpoints'])

    # Check if to resume from a checkpoint or a descriptor file
    if resume.endswith('.pth'):
        eval_approach.load_pretrained_model(resume)

    
    loop_range = list(range(0,120,1))
    eval_approach.load_hortov2_data()
    
    eval_approach.run(loop_range=loop_range)

    eval_approach.save_descriptors()
    eval_approach.save_params()
    eval_approach.save_predictions_pkl()
    eval_approach.save_results_csv()