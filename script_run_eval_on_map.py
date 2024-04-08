
import os

full_cap = '--epoch 300'


args = [#f'--network PointNetVLAD {local}',
        #f'--network PointNetORCHNetVLADSPoCLearned {local}',
        #f'--network PointNetORCHNetVLADSPoCMaxPooling {local}',
        #f'--network PointNetORCHNetMaxPooling {local}',
        #f'--network PointNetORCHNet {local}',
        #f'--network PointNetGeM {local}',
        #f'--network PointNetMAC  {local}',
        f'--network PointNetSPoC {local}',
        #f'--network overlap_transformer  {local}  --modality bev',
        f'--network LOGG3D {local}',
]

losses = ['LazyTripletLoss']

density = ['10000']


resume = 'saved_model_data/RAL/triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl/10000/PointNetGAP-LazyTripletLoss_L2-segment_lossM0.5/0.842@1'


test_sequrnces = [
        'ON22',
        'OJ22',
        'OJ23'
]


for seq in test_sequrnces:
        test_seq = '--val_set ' + seq
        func_arg = arg + ' ' +test_seq  #+ ' ' +  experiment +  ' ' + full_cap
        #print(func_arg)
        os.system('python3 tools/run_eval_on_map.py ' + func_arg)