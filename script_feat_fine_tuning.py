
import os

full_cap = '--epoch 50'
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

feat_dim = [16,32,64,128,256,512]


#density = ['500','1000','5000','10000','20000','30000']
density = ['10000']
experiment = f'-e cross_validation/final_tuning'

test_sequrnces = [
        'orchards/sum22/extracted',
        'orchards/june23/extracted',
        'orchards/aut22/extracted',
        'strawberry/june23/extracted'
]

for arg in args:
        for seq in test_sequrnces:
                for dim in feat_dim:
                        feat_input = "--feat_dim {}".format(str(dim))
                        experiment = f'-e cross_validation/final_tuning/featdim{dim}'
                        test_seq = '--test_seq ' + seq
                        func_arg = arg +  ' ' +  experiment +  ' ' + full_cap + ' ' + feat_input + ' ' + test_seq
                        #print(func_arg)
                        os.system('python3 train_knn.py ' + func_arg)