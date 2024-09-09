
import os



args = [#f'--network PointNetPGAP',
        #f'--network PointNetPGAPLoss',
        #f'--network PointNetVLADLoss',
        #f'--network PointNetVLAD',
        #f'--network overlap_transformer',
        f'--network LOGG3D',
]
#resume = 'saved_model_data/RAL/triplet/ground_truth_ar0.5m_nr10m_pr2m.pkl/10000/PointNetGAP-LazyTripletLoss_L2-segment_lossM0.5/0.842@1'


test_sequrnces = [
        '00',
        '02',
        '05',
        '06',
        '08',
]


for seq in test_sequrnces:
        test_seq = '--val_set ' + seq
        # Run ground truth 
        #os.system('python3 tools/run_eval_on_map.py ' + test_seq + ' --ground_truth 1')
        for net in args:
                
                func_arg = net + ' ' + test_seq  #+ ' show_plot 0'
                #print(func_arg)
                os.system('python3 tools/run_eval_on_map.py ' + func_arg)
        