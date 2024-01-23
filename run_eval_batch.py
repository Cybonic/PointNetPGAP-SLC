
import os

full_cap = '--epoch 300'

chkpt_root_A = '/home/deep/workspace/orchnet/v2/aa-0.5/checkpoints'
run = 'final@range1'

local = f'-e cross_validation/{run}  --chkpt_root {chkpt_root_A}'


args = [f'--network PointNetVLAD {local}',
        #f'--network PointNetORCHNetVLADSPoCLearned {local}',
        #f'--network PointNetORCHNetVLADSPoCMaxPooling {local}',
        #f'--network PointNetORCHNet {local}',
        #f'--network PointNetGeM {local}',
        #f'--network PointNetMAC  {local}',
        #f'--network PointNetSPoC {local}',
        #f'--network overlap_transformer  {local}  --modality bev',
        #f'--network LOGG3D {local}',
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
                os.system('python3 eval_knn.py ' + func_arg)