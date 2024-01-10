
import os

full_cap = '--epoch 80'
args = [#'--network PointNetVLAD',
        #'--network PointNetSOP',
        '--network LOGG3D',
        
        #'--network PointNetGeM',
        #'--network PointNetMAC',
        #'--network PointNetSPoC',
        #'--network overlap_transformer --modality bev',

        #'--network PointNetORCHNet',
        #'--network ResNet50ORCHNet --modality bev'
        #'--network ResNet50ORCHNetMaxPooling --modality bev',
        #'--network ResNet50GeM --modality bev',
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

evaluation_type = "cross_validation"
experiment = f'-e {evaluation_type}/iros24'

resume  = '--resume none'

test_sequrnces = [
        #'--val_set orchards/sum22/extracted --dataset uk',
        #'--val_set  orchards/june23/extracted --dataset uk',
        #'--val_set orchards/aut22/extracted --dataset uk',
        #'--val_set strawberry/june23/extracted --dataset uk',
        '--val_set e3/extracted --dataset GreenHouse'
]


for seq in test_sequrnces:
        for arg in args:
        
                test_seq =  seq
                model_evaluation = f'--model_evaluation {evaluation_type}' 
                func_arg = arg + ' ' + '--device cuda'+ ' ' + test_seq + ' ' +  experiment +  ' ' + full_cap + ' ' + model_evaluation + ' ' + resume
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg)