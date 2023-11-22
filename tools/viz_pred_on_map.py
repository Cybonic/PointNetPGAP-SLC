from dataloader.ORCHARDS import *

import yaml
import matplotlib.pyplot as plt

from utils.viz import myplot


if __name__=='__main__':
    session = 'orchard-uk.yaml'
    session_cfg_file = os.path.join('sessions', session)
 
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    root = SESSION['root']
    sequence = 'autumn'
    
    ground_truth = {
        'pos_range': 10, # Loop Threshold [m]
        'neg_range': 17,
        'num_neg': 20,
        'num_pos': 50,
        'warmupitrs': 600, # Number of frames to ignore at the beguinning
        'roi': 500
     }
    loader = ORCHARDSEval(root=root,dataset='',sequence=sequence,sync=True,modality='pcl',ground_truth=ground_truth)
    if sequence == 'summer':
        GT_LINES = SUMMER
    else:
        GT_LINES = AUTUMN
    poses  = loader.get_pose()
    anchor = loader.get_anchor_idx()
    map    = loader.get_map_idx()
    table  = loader.get_GT_Map() 

    true_loop = np.array([np.where(line==1)[0] for line in table])

    positives = true_loop[anchor]

    mplot = myplot(delay=0.5)
    mplot.init_plot(poses[:,0],poses[:,1],s = 10, c = 'whitesmoke')
    mplot.xlabel('m')
    mplot.ylabel('m')

    # gt_line_labels = loader.get_GT_Map()
    c = np.array(['r','b','y','k','g','m'])
    color = np.array(['whitesmoke']*poses.shape[0])
    scale = np.ones(poses.shape[0])*10

    indices = np.array(range(poses.shape[0]-1))
    #positives = []
    #anchors = []
    ROI = indices[2:]
    pos_range = 10
    roi = 500
    anchors = []
    positives = []
    pos_idx = []
    # for  i in : indices[roi+2:]
    # ASEGMENTS = ORCHARDS.ASEGMENTS
    # SUMMER = [1900,2610,3115]:
    for  i in [1900,2570,2950]:
        #i = 2700
        _map_   = poses[:i,:]
        pose    = poses[i,:].reshape((1,-1))
        
        pos_idx = true_loop[i]
        #pos = true_loop[i]
        if len(pos_idx)>0:
            n,c = poses.shape
            pa = poses[i].reshape(-1,c)
            pp = poses[pos_idx].reshape(-1,c)
            
            an_labels, an_point_idx = get_roi_points(pa,GT_LINES)
            #alabel = list(line_paa.keys())
            pos_labels, pos_point_idx = get_roi_points(pp,GT_LINES)
            # plabel = np.array(list(line_ppa.keys()))
            try:
                boolean_sg = np.where(an_labels[0] == pos_labels)[0]
                if len(boolean_sg):
                    pos = np.array([pos_idx[pos_point_idx[idx]] for idx in boolean_sg])
                    positives.extend(pos.flatten())
                    anchors.append(i)
            except:
                print("error")
            


        # Generate a colorize the head
        color = np.array(['k' for ii in range(0,i+1)])
        scale = np.array([5 for ii in range(0,i+1)])
        # Colorize the N smallest samples
       
        color[positives] = 'b'
        color[anchors] = 'r'
        scale[positives] = 100
        scale[anchors] = 100

        #if i % 20 == 0:
        mplot.update_plot(poses[:i+1,1],poses[:i+1,0],offset=2,zoom=0,color=color,scale=scale)


    print("The End")





