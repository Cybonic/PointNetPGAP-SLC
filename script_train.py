
import os
import time

# Define the number of epochs
epochs = 70
# Define the path to the checkpoints
# Define the path to the dataset

# Required to run on two different machines
hostname = os.popen('hostname').read().strip()
print(f"*** Hostname: {hostname}\n")
if hostname == 'tiago-deep':
        dataset_root = '/home/tiago/workspace/DATASET'
else:
        dataset_root = '/home/tbarros/workspace/DATASET'


TRAIN_FLAG = 1
# Path to save the predictions
save_path  = 'RALv3'

# Define the number of points
density = '10000'

EXPERIMENT_NAME = 'RALv3_kitti_test'
EXPERIMENT_NAME = 'Thesis_full'

EVAL_PROTOCOL = "cross_validation" # cross_domain

input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1'

#test_sequences_kitti = ['00','02','05','06','08']
#test_sequences_kitti = ['02','05','06','08']
test_sequences_horto = ['ON23','OJ22','OJ23','ON22','SJ23','GTJ23']

stages = [#'PointNetPGAP',
          #'PointNetPGAPLoss',
          #'PointNetVLAD',
          #'PointNetVLADLoss',
          'SPVSoAP3D',
          'SPVSoAP3DLoss',
          #'LOGG3D',
          #'LOGG3DLoss',
          #'overlap_transformer', 
          'overlap_transformerLoss',
          #'PointNetGeM',
          #'PointNetGeMLoss',
          #'PointNetMAC',
          #'PointNetMACLoss'
          ]

test_batchsize = [
                  16,
                  16,
                  16,
                  16,
                  16,
                  15
                  ] # 14 is the maximum batch size for GTJ23
eval_windows = [
        600,
        600,
        600,
        600,
        600,
        100, # 100 is the maximum window size for GTJ23
] 

checkpoint = f"checkpoints/RALv3_kittiv2/triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl/10000/00"
#time.sleep(1000)
for stage_conf in stages:
        for seq,testb,window in zip(test_sequences_horto,test_batchsize,eval_windows):
                for alpha in [10000]:
                #for alpha in [100,500,1000,3000,5000,10000,15000,20000,30000]:
                        func_arg = [
                                f'--network {stage_conf}', # Network
                                f'--train {TRAIN_FLAG}', # Train or test
                                f'--dataset_root {dataset_root}', # path to Dataset 
                                '--resume best_model', # [best_model, last_model]
                                #f'--resume {checkpoint}/{stage_conf}-LazyTripletLoss_L2-segment_loss-m0.5/best_model.pth', # [best_model, last_model]
                                #f'--resume {checkpoint}/{stage_conf}-LazyTripletLoss_L2-segment_loss-m0.5/checkpoint.pth', # [best_model, last_model]
                                #f'--resume {checkpoint}/{stage_conf}-LazyTripletLoss_L2/best_model.pth', # [best_model, last_model]
                                #f'--resume {checkpoint}/{stage_conf}-LazyTripletLoss_L2/checkpoint.pth', # [best_model, last_model]
                                f'--val_set {seq}',
                                f'--memory RAM' if TRAIN_FLAG == 1 else '--memory DISK', # [DISK, RAM] 
                                '--device cuda', # Device
                                f'--save_predictions {save_path}', # Save predictions
                                f'--epochs {epochs}',
                                f'--max_points {alpha}',
                                f'--experiment {EXPERIMENT_NAME}', 
                                #f'--feat_dim 16',
                                f'--eval_batch_size {testb}',
                                f'--mini_batch_size {1000}',
                                f'--loss_alpha 0.5',
                                f'--eval_roi_window {window}',
                                f'--eval_protocol {EVAL_PROTOCOL}',
                                input_preprocessing
                        ]
                        
                        func_arg_str = ' '.join(func_arg)        
                        print(func_arg_str)
                        os.system('python3 train_knn.py ' + func_arg_str)
                        
                        if TRAIN_FLAG == 1:
                                pass
                                time.sleep(600)