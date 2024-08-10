
import os

# Define the path to the dataset
dataset_root = '/home/tbarros/workspace/DATASET'

test_sequences = ['00',
                  '02',
                  '05',
                  '06',
                  '08',
                  #'GTJ23'
                ]

windows = [600,600,600,600,600,600]
resume_root = "/home/tbarros/workspace/pointnetgap-RAL/RALv3/kitti_predictions"

for seq,win in zip(test_sequences,windows):
        for network in ['PointNetPGAP']:
                func_arg = [ 
                        f'--dataset_root {dataset_root}', # path to Dataset 
                        f'--dataset HORTO-3DLM',
                        f'--val_set {seq}',
                        f'--resume {resume_root}/#{network}/eval-{seq}/descriptors.torch',
                        '--memory DISK', # [DISK, RAM] 
                        '--device cuda', # Device
                        f'--network {network}', # Network
                        f'--eval_roi_window {win}',
                        ]
                
                func_arg_str = ' '.join(func_arg)
                os.system('python3 eval_knn.py ' + func_arg_str)