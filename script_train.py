
import os
import time

# Define the number of epochs
epochs = 200
# Define the path to the checkpoints
# Define the path to the dataset
#dataset_root = '/home/tiago/workspace/DATASET'
dataset_root = '/home/tbarros/workspace/DATASET'

# Path to save the predictions
save_path  = 'RALv3'

# Define the number of points
density = '10000'

EXPERIMENT_NAME = 'RALv3'

input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1'

test_sequences = ['00','02','05','06','08']

stages = [#'PointNetPGAP',
          'PointNetPGAPLoss',
          #'PointNetVLAD',
          #'PointNetVLADLoss',
          #'LOGG3D',
          #'LOGG3DLoss',
          #'overlap_transformer', 
          #'overlap_transformerLoss',
          #'PointNetGeM',
          #'PointNetGeMLoss',
          #'PointNetMAC',
          #'PointNetMACLoss'
          ]

test_batchsize = [
                  15,
                  15,
                  16,
                  15,
                  15,
                  #15
                  ] # 14 is the maximum batch size for GTJ23
eval_windows = [
        600,
        600,
        600,
        600,
        600,
        #100, # 100 is the maximum window size for GTJ23
] 

checkpoint = "checkpoints/thesis/triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl/10000/OJ22"

for stage_conf in stages:
        #time.sleep(600)
        for seq,testb,window in zip(test_sequences,test_batchsize,eval_windows):
                for alpha in [10000]:
                #for alpha in [100,500,1000,3000,5000,10000,15000,20000,30000]:
                        func_arg = [
                                f'--network {stage_conf}', # Network
                                '--train 1', # Train or test
                                f'--dataset_root {dataset_root}', # path to Dataset 
                                #'--resume best_model', # [best_model, last_model]
                                f'--resume best_model', # [best_model, last_model]
                                f'--val_set {seq}',
                                '--memory RAM', # [DISK, RAM] 
                                '--device cuda', # Device
                                f'--save_predictions {save_path}', # Save predictions
                                f'--epochs {epochs}',
                                f'--max_points {alpha}',
                                f'--experiment {EXPERIMENT_NAME}', 
                                #f'--feat_dim 16',
                                f'--eval_batch_size {testb}',
                                f'--mini_batch_size {1000}',
                                f'--loss_alpha 0.5',
                                #f'--eval_roi_window {window}',
                                input_preprocessing
                        ]
                        
                        func_arg_str = ' '.join(func_arg)        
                        print(func_arg_str)
                        os.system('python3 train_knn.py ' + func_arg_str)