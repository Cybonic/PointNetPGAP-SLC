
import os

chkpt = "/home/tiago/workspace/pointnetgap-RAL/checkpoints"

root_dataset = "/home/tiago/workspace/DATASET"


models = ['PointNetGAP']

losses = ['LazyTripletLoss']

density    = ['10000']
experiment = "publishingtest"

test_sequences = [
        #'OJ22',
        #'OJ23',
        #'ON22',
        'SJ23'
]


for seq in test_sequences:
        for model in models:
                resume = os.path.join(chkpt,seq,model+'.pth')
                
                func_arg = [f'--val_set {seq}',
                            f'-e {experiment}',
                            f'--chkpt_root {chkpt}',
                            f'--network {model}',
                            f'--resume {resume}',
                            f'--dataset_root {root_dataset}',
                ]
                
                func_arg_str = ' '.join(func_arg)
                            
                os.system('python3 eval_knn.py ' + func_arg_str)