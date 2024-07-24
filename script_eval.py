
import os

# Define the path to the dataset
dataset_root = '/home/tbarros/workspace/DATASET'

test_sequences = ['SJ23',
                  'ON22',
                  'OJ22',
                  'ON23',
                  'OJ23',
                  'GTJ23'
                ]

resume_root = "/home/tbarros/workspace/pointnetgap-RAL/RALv3/predictions"

for seq in test_sequences:
        for network in ['PointNetPGAP']:
                func_arg = [ 
                        f'--dataset_root {dataset_root}', # path to Dataset 
                        f'--val_set {seq}',
                        f'--resume {resume_root}/#{network}-LazyTripletLoss_L2/eval-{seq}/descriptors.torch',
                        '--memory DISK', # [DISK, RAM] 
                        '--device cuda', # Device
                        f'--network {network}', # Network
                        ]
                
                func_arg_str = ' '.join(func_arg)
                os.system('python3 eval_knn.py ' + func_arg_str)