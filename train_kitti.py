
import os

full_cap = '--epoch 30'
args = [#'--network PointNetVLAD',
        '--network PointNet_ORCHNet',
        '--network ResNet50_ORCHNet --modality bev',
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
experiment = f'-e test_eval_data/30m' #cross_validation/final_tuning'

for loss_func in losses:
        loss =  f'--loss {loss_func}'
        for arg in args:
                func_arg = arg + ' ' + loss +  ' ' +  experiment +  ' ' + full_cap
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg)