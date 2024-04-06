#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import yaml
import os
import torch 

#from networks.orchnet import *
from trainer import Trainer
from pipeline_factory import model_handler,dataloader_handler
import numpy as np

# To force deterministic algorithms
# On terminal run the following command to set the environment variable
# export CUBLAS_WORKSPACE_CONFIG=":4096:8"
# os.system('export CUBLAS_WORKSPACE_CONFIG=:4096:8')
#torch.use_deterministic_algorithms(True)
  
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--train',
        type=int,
        required=False,
        default=1,
        help='Train the model.Default is True. If False, the model will be evaluated.'
    )
    
    parser.add_argument(
        '--dataset_root',
        type=str,
        required=False,
        default='/home/tiago/workspace/DATASET',
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--network', '-m',
        type=str,
        required=False,
        default='PointNetHGAPLoss',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=False,
        default='RAL',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--resume', '-r',
        type=str,
        required=False,
        default='None',
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
        '--epochs',
        type=int,
        required=False,
        default=100,
        help='Directory to get the trained model.'
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
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cpu',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=5,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--eval_batch_size',
        type=int,
        required=False,
        default=10,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--mini_batch_size',
        type=int,
        required=False,
        default=20000, #  Max size (based on the negatives)
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--loss',
        type=str,
        required=False,
        default = 'LazyTripletLoss',
        choices=['LazyTripletLoss'],
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
        help='number of features.'
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
        '--session',
        type=str,required=False,
        default = "ukfrpt",
    )
    
    parser.add_argument(
        '--augmentation',
        type=float,
        required=False,
        default = 1,
    )
    
    parser.add_argument(
        '--shuffle_points',
        type=float,
        required=False,
        default = 1,
    )
    parser.add_argument(
        '--pcl_norm',
        type=float,
        required=False,
        default = 0,
    )
    parser.add_argument(
        '--eval_roi_window',
        type=float,
        required=False,
        default = 600,
    )
    parser.add_argument(
        '--stages',
        type=str,
        required=False,
        default = '111',
    )
    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    
    SESSION['pcl_norm'] = FLAGS.pcl_norm
    # Update config file with new settings
    SESSION['experiment'] = FLAGS.experiment
    SESSION['trainer']['minibatch_size']  = FLAGS.mini_batch_size
    SESSION['trainer']['feat_dim']  = FLAGS.feat_dim
    SESSION['aug']  = FLAGS.augmentation
    # Define evaluation mode: cross_validation or split
    SESSION['model_evaluation'] = FLAGS.model_evaluation
    
   
    SESSION['train_loader']['triplet_file'] = FLAGS.triplet_file
    SESSION['train_loader']['augmentation'] = FLAGS.augmentation
    SESSION['train_loader']['shuffle_points'] = FLAGS.shuffle_points
    
    SESSION['val_loader']['batch_size'] = FLAGS.eval_batch_size
    SESSION['val_loader']['ground_truth_file'] = FLAGS.eval_file
    SESSION['val_loader']['augmentation'] = False
    
    
    SESSION['trainer']['epochs'] =  FLAGS.epochs
    SESSION['loss']['type'] = FLAGS.loss
    SESSION['max_points']= FLAGS.max_points
    SESSION['memory']= FLAGS.memory
    
    SESSION['monitor_range']   = FLAGS.loop_range
    SESSION['eval_roi_window'] = FLAGS.eval_roi_window


    print("----------")
    print("Saving Predictions: %d"%FLAGS.save_predictions)
    # print("Root: ", SESSION['root'])
    print("\n======= TRAIN LOADER =======")
    # print("Dataset  : ", SESSION['train_loader']['data']['dataset'])  print("Sequence : ", SESSION['train_loader']['data']['sequence'])
    print("Max Points: " + str(SESSION['max_points']))
    print("Triplet Data File: " + str(FLAGS.triplet_file))
    print("Augmentation: " + str(SESSION['train_loader']['augmentation']))
    print("Batch Size : ", str(SESSION['train_loader']['batch_size']))
    print("MiniBatch Size: ", str(SESSION['trainer']['minibatch_size']))
    
    print("\n======= VAL LOADER =======")
    print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
    print("Max Points: " + str(SESSION['max_points']))
    print("Eval Data File: " + str(FLAGS.eval_file))
    print("Augmentation: " + str(SESSION['val_loader']['augmentation']))
    print("Eval window : " + str(SESSION['eval_roi_window']))
    
    
    print("\n========== MODEL =========")
    print("Backbone : ", FLAGS.network)
    print("Resume: ",  FLAGS.resume )
    print("Loss: ",FLAGS.loss)
    print("MiniBatch Size: ", str(SESSION['trainer']['minibatch_size']))
    
    
    print("\n==========================")
    print(f'Memory: {FLAGS.memory}')
    print(f'Device: {FLAGS.device}')
    print("Loss: %s" %(SESSION['loss']['type']))
    print("Experiment: %s" %(FLAGS.experiment))
    print("Max epochs: %s" %(FLAGS.epochs))
    print("PCL Norm: %s" %(FLAGS.pcl_norm))
    #print("Modality: %s" %(model_param['modality']))
    print("----------\n")

    # For repeatability
    torch.manual_seed(0)
    np.random.seed(0)
    
    # Build the model and the loader
    model_ = model_handler(FLAGS.network,
                            num_points=SESSION['max_points'],
                            output_dim =256,
                            feat_dim  = FLAGS.feat_dim,
                            device    = FLAGS.device,
                            loss      = SESSION['loss'],
                            trainer = SESSION['trainer'],
                            stage_1 = bool(int(FLAGS.stages[0])),
                            stage_2 = bool(int(FLAGS.stages[1])),
                            stage_3 = bool(int(FLAGS.stages[2]))
                            )
    
    loader = dataloader_handler(FLAGS.dataset_root,
                                FLAGS.network,
                                FLAGS.dataset,
                                FLAGS.val_set,
                                SESSION,
                                roi = FLAGS.roi,
                                pcl_norm = FLAGS.pcl_norm)

    run_name = {'dataset': '-'.join(str(SESSION['val_loader']['sequence'][0]).split('/')),
                'experiment':os.path.join(FLAGS.experiment,FLAGS.triplet_file,str(FLAGS.max_points)), 
                'model': str(model_)
            }

    trainer = Trainer(
            model        = model_,
            train_loader = loader.get_train_loader(),
            val_loader   = loader.get_val_loader(),
            resume = FLAGS.resume,
            config = SESSION,
            device = FLAGS.device,
            run_name = run_name,
            train_epoch_zero = True,
            monitor_range = SESSION['monitor_range'],
            roi_window    = FLAGS.eval_roi_window,
            eval_protocol = 'place',
            debug = False
            )
    

    if FLAGS.train:
        loop_range = [1,5,10,20]
        best_model_filename = trainer.Train(train_batch=FLAGS.batch_size,loop_range=loop_range)
    
    
    if FLAGS.save_predictions:
        
        best_model_filename = trainer.save_best_model_filename
        # Generate descriptors, predictions and performance for the best weights
        trainer.eval_approach.load_pretrained_model(best_model_filename)
        loop_range = list(range(0,120,1))
        trainer.eval_approach.run(loop_range=loop_range)

        trainer.eval_approach.save_params()
        trainer.eval_approach.save_descriptors()
        trainer.eval_approach.save_predictions_cv()
        trainer.eval_approach.save_results_cv()
