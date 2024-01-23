
import os

full_cap = '--epoch 80'
args = [#'--network PointNetVLAD',
        '--network PointNetSOP',
        #'--network LOGG3D',
        
        #'--network PointNetGeM',
        #'--network PointNetMAC',
        #'--network PointNetSPoC',
        #'--network overlap_transformer --modality bev',

        #'--network PointNetORCHNet',
        #'--network ResNet50ORCHNet --modality bev'
        #'--network ResNet50ORCHNetMaxPooling --modality bev',
        #'--network ResNet50GeM --modality bev',
        #' --network overlap_transformer',
]

#losses = ['PositiveLoss','LazyTripletLoss','LazyQuadrupletLoss']
#losses = ['LazyTripletLoss','LazyQuadrupletLoss']
losses = ['LazyTripletLoss']

#density = ['500','1000','5000','10000','20000','30000']
density = ['10000']

evaluation_type = "cross_validation"
experiment = f'-e {evaluation_type}/final@range1'

resume  = '--resume best_model'

test_sequrnces = [
        'orchards/sum22/extracted',
        'orchards/june23/extracted',
        'orchards/aut22/extracted',
        'strawberry/june23/extracted'
]


for seq in test_sequrnces:
        for arg in args:
        
                test_seq = '--val_set ' + seq
                model_evaluation = f'--model_evaluation {evaluation_type}' 
                func_arg = arg + ' ' + '--device cuda'+ ' ' + test_seq + ' ' +  experiment +  ' ' + full_cap + ' ' + model_evaluation + ' ' + resume
                #print(func_arg)
                os.system('python3 train_knn.py ' + func_arg)