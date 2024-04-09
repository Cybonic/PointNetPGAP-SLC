
import os


# Define the path to the checkpoints
chkpt_root = '/home/tiago/workspace/pointnetgap-RAL/checkpoints' # Path to the checkpoints or descriptors
# Define the path to the dataset
dataset_root = '/home/tiago/workspace/DATASET'

resume  = "PointNetGAP.pth" # choice [checkpoints.pth, descriptors.torch]

# Path to save the predictions
save_path  = 'predictions'

test_sequences = ['OJ22','OJ23','ON22','SJ23']

resume_root = "/home/tiago/workspace/pointnetgap-RAL/RALv2/predictions_RALv1"
for seq in test_sequences:
        for network in ['overlap_transformer','PointNetGAP','PointNetGeM','PointNetMAC','PointNetVLAD']:
                func_arg = [ 
                        f'--dataset_root {dataset_root}', # path to Dataset 
                        f'--val_set {seq}',
                        f'--resume {resume_root}/{network}/{seq}/descriptors.torch',
                        '--memory DISK', # [DISK, RAM] 
                        '--device cuda', # Device
                        f'--network {network}', # Network
                        #f'--save_predictions {save_path}', # Save predictions
                        ]
                
                func_arg_str = ' '.join(func_arg)
                os.system('python3 eval_knn.py ' + func_arg_str)