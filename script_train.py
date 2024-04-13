
import os


# Define the number of epochs
epochs = 100
# Define the path to the checkpoints
# Define the path to the dataset
#dataset_root = '/home/tiago/workspace/DATASET'
dataset_root = '/home/tbarros/workspace/DATASET'

# Path to save the predictions
save_path  = 'predictions/RAL'

# Define the number of points
density = '10000'

input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1'

test_sequences = ['OJ22','OJ23','SJ23','ON22']

stages = ['LOGG3D','PointNetVLAD','overlap_transformer',]
for stage_conf in stages:
        for seq in test_sequences:
                func_arg = [
                        f'--network {stage_conf}', # Network
                        '--train 0', # Train or test
                        f'--dataset_root {dataset_root}', # path to Dataset 
                        '--resume best_model', # [best_model, last_model]
                        f'--val_set {seq}',
                        '--memory RAM', # [DISK, RAM] 
                        '--device cuda', # Device
                        f'--save_predictions {save_path}', # Save predictions
                        f'--epochs {epochs}',
                        #f'--stages {stage_conf}',
                        f'--experiment MSGAP', 
                        f'--feat_dim 1024',
                        f'--eval_batch_size {10}',
                        input_preprocessing
                ]
                        
                func_arg_str = ' '.join(func_arg)        
                
                os.system('python3 train_knn.py ' + func_arg_str)