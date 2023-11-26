
import os

full_cap = '--epoch 100'
args = [#'--network PointNetVLAD',
        #'--network PointNet_ORCHNet',
        #'--network ResNet50_ORCHNet --modality bev',
        '--network ResNet50ORCHNetMaxPooling --modality bev',
        '--network PointNetORCHNetMaxPooling',
        
        #'--network ResNet50GeM --modality bev',
        #'--network PointNetGeM',
        #'--network overlap_transformer'

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
experiment = f'-e cross_validation/final_tuning'


test_sequrnces = [
        'orchards/sum22/extracted',
        'orchards/june23/extracted',
        'orchards/aut22/extracted',
        'strawberry/june23/extracted'
]

for seq in test_sequrnces:
        for arg in args:
                test_seq = '--val_set ' + seq
                func_arg = arg + ' ' +test_seq + ' ' +  experiment +  ' ' + full_cap
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg)