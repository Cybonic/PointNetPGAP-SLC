
import os


# Define the number of epochs
epochs = 100
# Define the path to the checkpoints
# Define the path to the dataset
# dataset_root = '/home/tiago/workspace/DATASET'
dataset_root = '/home/tiago/workspace/DATASET'

# Path to save the predictions
save_path  = 'predictions/RAL'

# Define the number of points
density = '10000'

input_preprocessing = ' --roi 0 --augmentation 1 --shuffle_points 1'

test_sequences = ['OJ22','OJ23','ON22', 'SJ23']

for seq in test_sequences:
        func_arg = [
                '--train 0', # Train or test
                f'--dataset_root {dataset_root}', # path to Dataset 
                '--resume best_model', # [best_model, last_model]
                f'--val_set {seq}',
                '--memory RAM', # [DISK, RAM] 
                '--device cuda', # Device
                f'--save_predictions {save_path}', # Save predictions
                f'--epochs {epochs}',
                input_preprocessing
        ]
                
        func_arg_str = ' '.join(func_arg)        
        
        os.system('python3 train_knn.py ' + func_arg_str)