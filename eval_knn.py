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
        '--network', '-m',
        type=str,
        required=False,
        default='PointNetVLAD',
        choices=['PointNetORCHNetMaxPooling',
                'PointNetORCHNetVLADSPoCLearned',
                'PointNetORCHNetVLADSPoCMaxPooling',   
                'PointNetORCHNet',
                 'PointNetVLAD',
                 'LOGG3D',
                 'PointNetORCHNet',
                 'ResNet50ORCHNet',
                 'ResNet50GeM',
                 'PointNetGeM',
                 'ResNet50MAC',
                 'PointNetMAC',
                 'ResNet50SPoC',
                 'PointNetSPoC',
                 'overlap_transformer'],
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=False,
        default='cross_validation/final@range1',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        required=False,
        default='cross_validation/ON22/PointNetVLAD.pth',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--memory',
        type=str,
        required=False,
        default='DISK',
        choices=['DISK','RAM'],
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--epoch',
        type=int,
        required=False,
        default=2,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='uk', # uk
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=20,
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--max_points',
        type=int,
        required=False,
        default = 10000,
        help='sampling points.'
    )
    parser.add_argument(
        '--feat_dim',
        type=int,
        required=False,
        default = 1024,
        help='sampling points.'
    )
    parser.add_argument(
        '--modality',
        type=str,
        required=False,
        default = "pcl",
        help='sampling points.'
    )

    parser.add_argument(
        '--triplet_file',
        type=str,
        required=False,
        default = "triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl",
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
        '--loop_range',
        type=float,
        required=False,
        default = 10,
        help='sampling points.'
    )

    parser.add_argument(
        '--save_predictions',
        type=bool,
        required=False,
        default = True,
        help='sampling points.'
    )
    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default = 'orchards/aut22/extracted',
    )

    parser.add_argument(
        '--roi',
        type=float,
        required=False,
        default = 0,
    )
    parser.add_argument(
        '--model_evaluation',
        type=str,
        required=False,
        default = "cross_validation",
        choices = ["cross_validation"]
    )
    parser.add_argument(
        '--chkpt_root',
        type=str,
        required=False,
        #default = "/home/deep/workspace/orchnet/v2/aa-0.5/checkpoints"
        default = "/home/deep/workspace/orchnet/v2/paper_checkpoints"
    )
    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default = "uk",
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session.lower() + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))


    SESSION['trainer']['save_dir'] =  FLAGS.chkpt_root
    # Update config file with new settings
    SESSION['experiment'] = FLAGS.experiment
    SESSION['modelwrapper']['minibatch_size']  = None
    SESSION['modelwrapper']['feat_dim']  = FLAGS.feat_dim
    
    # Define evaluation mode: cross_validation or split
    SESSION['model_evaluation'] = FLAGS.model_evaluation
    SESSION['train_loader']['sequence'] = SESSION['cross_validation'][FLAGS.val_set]
    SESSION['val_loader']['sequence']   = [FLAGS.val_set]
   
    SESSION['val_loader']['batch_size'] = FLAGS.batch_size
    SESSION['train_loader']['triplet_file'] = FLAGS.triplet_file
    SESSION['val_loader']['ground_truth_file'] = FLAGS.eval_file
    SESSION['trainer']['epochs'] =  FLAGS.epoch
    SESSION['loss']['type'] = None
    SESSION['max_points']= FLAGS.max_points
    SESSION['memory']= FLAGS.memory
    
    
    SESSION['loop_range'] = FLAGS.loop_range


    print("----------")
    print("Saving Predictions: %d"%FLAGS.save_predictions)
    # print("Root: ", SESSION['root'])
    print("\n======= TRAIN LOADER =======")
    # print("Dataset  : ", SESSION['train_loader']['data']['dataset'])  print("Sequence : ", SESSION['train_loader']['data']['sequence'])
    print("Max Points: " + str(SESSION['max_points']))
    print("Triplet Data File: " + str(FLAGS.triplet_file))
    print("\n======= VAL LOADER =======")
    print("Dataset  : ", SESSION['val_loader']['dataset'])
    print("Sequence : ", SESSION['val_loader']['sequence'])
    print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
    print("Max Points: " + str(SESSION['max_points']))
    print("\n========== MODEL =========")
    print("Backbone : ", FLAGS.network)
    print("Resume: ",  FLAGS.resume )
    print("MiniBatch Size: ", str(SESSION['modelwrapper']['minibatch_size']))
    print("\n==========================")
    print(f'Memory: {FLAGS.memory}')
    print(f'Device: {FLAGS.device}')
    print("Loss: %s" %(SESSION['loss']['type']))
    print("Experiment: %s" %(FLAGS.experiment))
    print("Max epochs: %s" %(FLAGS.epoch))
    #print("Modality: %s" %(model_param['modality']))
    print("----------\n")

    # For repeatability
    torch.manual_seed(0)
    np.random.seed(0)


    ###################################################################### 

    # The development has been made on different PCs, each has some custom settings
    # e.g the root path to the dataset;
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))

    root_dir = pc_config[device_name]
    
    # Build the model and the loader
    model = model_handler(FLAGS.network,
                            num_points=SESSION['max_points'],
                            output_dim=256,
                            feat_dim=FLAGS.feat_dim,
                            device = FLAGS.device,
                            loss = None, # Loss is required to build the model name correctly (change this in the future)
                            modelwrapper = SESSION['modelwrapper']
                            )

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)


    loader = dataloader_handler(root_dir,FLAGS.network,FLAGS.dataset,SESSION, roi = FLAGS.roi)

    # Hack
    # add dataset name at the begining
    dataset_name = '-'.join(str(SESSION['val_loader']['sequence'][0]).split('/'))
    dataset_name = FLAGS.dataset + '-' + dataset_name
    run_name = {'dataset': dataset_name,
                'experiment':os.path.join(FLAGS.experiment,FLAGS.triplet_file,str(FLAGS.max_points)), 
                'model': str(model)
            }

    trainer = Trainer(
            model        = model,
            train_loader = None,
            test_loader   = loader.get_test_loader(),
            resume = None,
            config = SESSION,
            device = FLAGS.device,
            run_name = run_name,
            train_epoch_zero = False,
            )
    
    loop_range = list(range(0,120,1))
    
    best_model_filename = os.path.join(FLAGS.chkpt_root,FLAGS.resume) 
    
    # Generate descriptors, predictions and performance for the best weights
    print(f'\nLoading best model: {best_model_filename}\n')
    trainer.eval_approach.load_pretrained_model(best_model_filename)

    trainer.eval_approach.run(loop_range=loop_range)
    
    save_to = "saved_model_data/final@range1"
    trainer.eval_approach.save_params(save_to)
    trainer.eval_approach.save_descriptors(save_to)
    trainer.eval_approach.save_predictions_cv(save_to)
    trainer.eval_approach.save_results_cv(save_to)
