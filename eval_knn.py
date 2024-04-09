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
import argparse
import yaml
import os
import torch 

#from networks.orchnet import *
from trainer import Trainer
from networks import contrastive
from utils import loss as losses
from pipeline_factory import model_handler,dataloader_handler
import numpy as np

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--dataset_root',type=str, required=False,
        default='/home/tiago/workspace/DATASET',
        help='Directory to the dataset root'
    )
    
    parser.add_argument(
        '--network', type=str,
        default='LOGG3D', help='model to be used'
    )

    parser.add_argument(
        '--experiment',type=str,
        default='RAL',
        help='Name of the experiment to be executed'
    )

    parser.add_argument(
        '--memory', type=str,
        default='DISK',
        choices=['DISK','RAM'],
        help='RAM: loads the dataset to the RAM memory first. DISK: loads the dataset on the fly from the disk'
    )

    parser.add_argument(
        '--device', type=str,
        default='cuda',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',type=int,
        default=10,
        help='Batch size'
    )
    
    parser.add_argument(
        '--max_points',type=int,
        default = 10000,
        help='sampling points.'
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        required=False,
        default = "eval/ground_truth_loop_range_10m.pkl",
        help='sampling points.'
    )

    parser.add_argument(
        '--monitor_loop_range',
        type=float,
        required=False,
        default = 10,
        help='loop range to monitor the performance.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='HORTO-3DLM', # uk
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default = 'ON22',
        help = 'Validation set'
    )

    parser.add_argument(
        '--roi',
        type=float,
        required=False,
        default = 0,
        help = 'Crop range [m] to crop the point cloud around the scan origin.'
    )
    
    parser.add_argument(
        '--resume', '-r',
        type=str,
        required=False,
        default='/home/tiago/workspace/pointnetgap-RAL/RALv2/predictions_RALv1/LOGG3D/ON22/descriptors.torch',
        help='Directory to get the trained model or descriptors.'
    )

    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default = "ukfrpt",
    )
    
    parser.add_argument(
        '--eval_roi_window',
        type=float,
        required=False,
        default = 600,
        help='Number of frames to ignore in imidaite vicinity of the query frame.'
    )
    
    parser.add_argument(
        '--eval_warmup_window',
        type=float,
        required=False,
        default = 100,
        help='Number of frames to ignore in the beginning of the sequence'
    )
    
    parser.add_argument(
        '--eval_protocol',
        type=str,
        required=False,
        choices=['place'],
        default = 'place',
    )
    
    parser.add_argument(
        '--save_predictions',
        type=str,
        required=False,
        default = 'saved_model_data',
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    SESSION['save_predictions'] = FLAGS.save_predictions
    # Update config file with new settings
    SESSION['experiment'] = FLAGS.experiment
    
    # Define evaluation mode: cross_validation or split
    SESSION['train_loader']['triplet_file'] = None
    
    # Update the validation loader
    SESSION['val_loader']['batch_size'] = FLAGS.batch_size
    SESSION['val_loader']['ground_truth_file'] = FLAGS.eval_file
    SESSION['val_loader']['augmentation'] = False
    
    # Update the model settings
    SESSION['roi'] = FLAGS.roi
    SESSION['max_points'] = FLAGS.max_points
    SESSION['memory']     = FLAGS.memory
    SESSION['monitor_range']   = FLAGS.monitor_loop_range
    SESSION['eval_roi_window'] = FLAGS.eval_roi_window
    SESSION['descriptor_size'] = 256
    SESSION['eval_warmup_window'] = FLAGS.eval_warmup_window
    SESSION['eval_protocol'] = FLAGS.eval_protocol
    SESSION['device'] = FLAGS.device


    print("----------")
    print("Saving Predictions: %s"%FLAGS.save_predictions)
    print("\n======= VAL LOADER =======")
    print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
    print("Max Points: " + str(SESSION['max_points']))
    print("\n========== MODEL =========")
    print("Backbone : ", FLAGS.network)
    print("Resume: ",  FLAGS.resume )
    #print("MiniBatch Size: ", str(SESSION['modelwrapper']['minibatch_size']))
    print("\n==========================")
    print(f'Eval Protocal: {FLAGS.eval_protocol}')
    print(f'Memory: {FLAGS.memory}')
    print(f'Device: {FLAGS.device}')
    print("Experiment: %s" %(FLAGS.experiment))
    print("----------\n")

    # For repeatability
    
    torch.manual_seed(0)
    np.random.seed(0)

    if os.path.isfile(FLAGS.resume):
        print("Resuming form %s"%FLAGS.resume)
        
        resume_struct= FLAGS.resume.split('/')
        assert FLAGS.val_set in resume_struct, "The resume file does not match the validation set"
        assert FLAGS.network in resume_struct, "The resume file does not match the network" 
    
    ###################################################################### 
    loader = dataloader_handler(FLAGS.dataset_root,
                                FLAGS.network,
                                FLAGS.dataset,
                                FLAGS.val_set,
                                SESSION, 
                                roi = FLAGS.roi, 
                                pcl_norm = False)

  

    from place_recognition import PlaceRecognition
    eval_approach = PlaceRecognition(   FLAGS.network,
                                        loader.get_val_loader(),
                                        1, # Internally it adds top-1% 
                                        None,
                                        roi_window = FLAGS.eval_roi_window,
                                        warmup_window = FLAGS.eval_warmup_window,
                                        device = FLAGS.device,
                                        eval_protocol = FLAGS.eval_protocol,
                                        logdir = FLAGS.experiment,
                                        monitor_range = FLAGS.monitor_loop_range,
                                        sim_func= 'L2'
                                        )

    
    # Define a set of loop ranges to be evaluated
    loop_range = list(range(0,120,1))
    
    # Check if the resume file exists
    assert os.path.exists(FLAGS.resume), "File not found %s"%FLAGS.resume 

    root_path = os.path.dirname(FLAGS.resume)
        
    if FLAGS.resume.split('/')[-1] == 'descriptors.torch':
        eval_approach.load_descriptors(FLAGS.resume)
    
    # Run the evaluation
    eval_approach.run(loop_range=loop_range)
    
    eval_approach.save_params()
    eval_approach.save_descriptors()
    eval_approach.save_predictions_pkl()
    eval_approach.save_results_csv()
