
import os



root = "/home/deep/workspace/orchnet/v2"

chkpt = os.path.join(root,'checkpoints')
dataset = os.path.join(root,'agro3d-rc')


models = ['PointNetVLAD',
        #f'--network PointNetGeM',
        #f'--network PointNetMAC',
        #f'--network PointNetSPoC',
        #f'--network overlap_transformer --modality bev',
        #f'--network LOGG3D',
]

losses = ['LazyTripletLoss']

density    = ['10000']
experiment = "publishingtest"

test_sequences = [
        'OJ22',
        'OJ23',
        'ON22',
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
                            f'--dataset_root {dataset}',
                ]
                
                func_arg_str = ' '.join(func_arg)
                            
                #print(func_arg)
                os.system('python3 eval_knn.py ' + func_arg_str)