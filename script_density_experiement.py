
import os

full_cap = '--epoch 50'
args = [#'--network PointNetVLAD',
        #'--network PointNet_ORCHNet',
        '--network ResNet50_ORCHNet',
        '--modality bev',
        #'--network ResNet50GeM',
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

density = ['10000']
#density = ['10000']
experiment = f'-e test_bev_roi'


        #loss =  f'--loss {loss_func}'
for arg in args:
        for n_points in density:
                density_arg = f'--max_points {n_points}'
                func_arg = arg + ' ' +  experiment +  ' ' + full_cap + ' ' + density_arg
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg)