
import os


# Define the path to the checkpoints
chkpt_root = '/home/tiago/workspace/pointnetgap-RAL/checkpoints' # Path to the checkpoints or descriptors
# Define the path to the dataset
dataset_root = '/home/tiago/workspace/DATASET'

resume  = "PointNetGAP.pth" # choice [checkpoints.pth, descriptors.torch]

# Path to save the predictions
save_path  = 'predictions'

test_sequences = ['OJ22','OJ23','ON22','SJ23']

for seq in test_sequences:
        func_arg = [ 
                f'--dataset_root {dataset_root}', # path to Dataset 
                f'--val_set {seq}',
                f'--resume {chkpt_root}/{seq}/{resume}',
                '--memory DISK', # [DISK, RAM] 
                '--device cuda', # Device
                f'--save_predictions {save_path}', # Save predictions
                ]
        
        func_arg_str = ' '.join(func_arg)
        os.system('python3 eval_knn.py ' + func_arg_str)