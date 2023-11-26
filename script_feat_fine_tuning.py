
import os

full_cap = '--epoch 100'
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

feat_dim = [1024]


#density = ['500','1000','5000','10000','20000','30000']
density = ['10000']
#experiment = f'-e cross_validation/final_tuning'

test_sequrnces = [
        'orchards/sum22/extracted',
        #'orchards/june23/extracted',
        #'orchards/aut22/extracted',
        #'strawberry/june23/extracted'
]

for arg in args:
        for seq in test_sequrnces:
                for dim in feat_dim:
                        for  roi in [10,20,30,40,50,60,100]:
                                feat_input = "--feat_dim {}".format(str(dim))
                                experiment = f'-e cross_validation/eval_roi/{roi}m/featdim{dim}'
                                test_seq = '--val_set ' + seq
                                roi_flag = "--roi " + str(roi)
                                func_arg = arg +  ' ' +  experiment +  ' ' + full_cap + ' ' + feat_input + ' ' + test_seq + ' ' + roi_flag
                                #print(func_arg)
                                os.system('python3 train_knn.py ' + func_arg)