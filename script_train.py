
import os


# Define the number of epochs
epochs = 400
# Define the path to the checkpoints
# Define the path to the dataset
dataset_root = '/home/tiago/workspace/DATASET'
#dataset_root = '/home/tbarros/workspace/DATASET'

# Path to save the predictions
save_path  = 'predictions/RAL'

# Define the number of points
density = '10000'

input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1'

test_sequences = ['SJ23','ON22','OJ22','ON23']

stages = ['PointNetPGAP']
for stage_conf in stages:
        for seq in test_sequences:
                for alpha in [10000]:
                #for alpha in [100,500,1000,3000,5000,10000,15000,20000,30000]:
                        func_arg = [
                                f'--network {stage_conf}', # Network
                                '--train 1', # Train or test
                                f'--dataset_root {dataset_root}', # path to Dataset 
                                '--resume best_model', # [best_model, last_model]
                                f'--val_set {seq}',
                                '--memory RAM', # [DISK, RAM] 
                                '--device cuda', # Device
                                f'--save_predictions {save_path}', # Save predictions
                                f'--epochs {epochs}',
                                f'--max_points {alpha}',
                                f'--experiment RAL/POINT_DENSITY', 
                                f'--feat_dim 16',
                                f'--eval_batch_size {15}',
                                f'--mini_batch_size {1000}',
                                f'--loss_alpha 0.5',
                                input_preprocessing
                        ]
                        
                        func_arg_str = ' '.join(func_arg)        
                        
                        os.system('python3 train_knn.py ' + func_arg_str)