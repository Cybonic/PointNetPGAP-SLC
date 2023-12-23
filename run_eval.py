
import os

full_cap = '--epoch 300'

chkpt_root_B = '/home/deep/Dropbox/SHARE/orchards-uk/code/place_recognition_models/checkpoints'
chkpt_root_A = '/home/deep/workspace/orchnet/v2/aa-0.5/checkpoints'
args = [f'--network PointNetVLAD  -e cross_validation/final@range1  --chkpt_root {chkpt_root_B}',
        #'--network PointNetORCHNet',
        #'--network ResNet50ORCHNet --modality bev'
        #'--network ResNet50ORCHNetMaxPooling --modality bev',
        
        f'--network PointNetORCHNet -e cross_validation/finalMyModels-no_augv2  --chkpt_root {chkpt_root_B}' ,
        
        #'--network ResNet50GeM --modality bev',
        f'--network PointNetGeM -e cross_validation/baselines  --chkpt_root {chkpt_root_A}',
        #'--network ResNet50MAC --modality bev',
        f'--network PointNetMAC  -e cross_validation/baselines --chkpt_root {chkpt_root_A}' ,
        #'--network ResNet50SPoC --modality bev',
        f'--network PointNetSPoC -e cross_validation/baselines --chkpt_root {chkpt_root_A}',
        f'--network overlap_transformer  --modality bev -e cross_validation/baselines --chkpt_root {chkpt_root_A}',
        f'--network LOGG3D -e cross_validation/baselines --chkpt_root {chkpt_root_A}',
        #'--network overlap_transformer --modality bev',

        #' --network overlap_transformer',
        #f'--memory RAM  --modality bev  --session kitti --model VLAD_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model SPoC_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model GeM_resnet50 ',
        #f'--memory RAM  --modality bev  --session kitti --model MuHA_resnet50',
]

#losses = ['PositiveLoss','LazyTripletLoss','LazyQuadrupletLoss']
#losses = ['LazyTripletLoss','LazyQuadrupletLoss']
losses = ['LazyTripletLoss']

#density = ['500','1000','5000','10000','20000','30000']
density = ['10000']
#experiment = f'-e cross_validation/finalMyModels-no_aug'


test_sequrnces = [
        'orchards/sum22/extracted',
        'orchards/june23/extracted',
        'orchards/aut22/extracted',
        'strawberry/june23/extracted'
]


for seq in test_sequrnces:
        for arg in args:
                test_seq = '--val_set ' + seq
                func_arg = arg + ' ' +test_seq  #+ ' ' +  experiment +  ' ' + full_cap
                #print(func_arg)
                os.system('python3 eval_knnv2.py ' + func_arg)