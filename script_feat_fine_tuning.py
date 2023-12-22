
import os

full_cap = '--epoch 100'
args = [#'--network PointNetVLAD',
        '--network PointNetORCHNet',
        #'--network ResNet50_ORCHNet --modality bev',
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
        #'orchards/sum22/extracted',
        #'orchards/june23/extracted',
        #'orchards/aut22/extracted',
        'strawberry/june23/extracted'
]
evaluation_type = "cross_validation"

for arg in args:
        for seq in test_sequrnces:
                for dim in feat_dim:
                        for  roi in [10,50,100,150,200]:
                                model_evaluation = f'--model_evaluation {evaluation_type}' 
                                #feat_input = "--feat_dim {}".format(str(dim))
                                experiment = f'-e {evaluation_type}/eval_roi/{roi}m'
                                test_seq = '--val_set ' + seq
                                roi_flag = "--roi " + str(roi)
                                func_arg = arg +  ' ' +  experiment +  ' ' + full_cap + ' '  + ' ' + test_seq + ' ' + roi_flag + ' ' + model_evaluation
                                #print(func_arg)
                                os.system('python3 train_knn.py ' + func_arg)