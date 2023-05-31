import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm import tqdm

path = os.path.dirname(os.path.realpath(__file__))
path = path.split(os.sep)
path = os.sep.join(path[:-1])
sys.path.append(path)

from dataloader.ORCHARDS import OrchardDataset
from dataloader.KITTI import KittiDataset
from utils.viz import myplot
import yaml
from dataloader.utils import load_dataset
#from scipy.spatial import distanc



def viz_overlap(xy, gt_loops, plot_flag = True, record_gif= False, file_name = 'anchor_positive_pair.gif'):

    indices = np.array(range(xy.shape[0]-1))

    if plot_flag == True:
        mplot = myplot(delay=0.2)
        mplot.init_plot(xy[:,0],xy[:,1],s = 10, c = 'whitesmoke')
        mplot.xlabel('m'), mplot.ylabel('m')
        if record_gif == True:
            mplot.record_gif(file_name)
    
    ROI = indices[2:]
    positives  = []
    anchors    = []
    pos_distro = []

    for i in ROI:
        idx = gt_loops[i]
        if len(idx)>0:
            positives.extend(idx)
            anchors.append(i)
            pos_distro.append(idx)
        # Generate a colorize the head
        color = np.array(['k' for ii in range(0,i+1)])
        scale = np.array([30 for ii in range(0,i+1)])
        # Colorize the N smallest samples
        color[anchors]   = 'r'
        color[positives] = 'b'
        scale[positives] = 100

        if plot_flag == True and i % 100 == 0:
            x = xy[:i,1]#[::-1]
            y = xy[:i,0]#[::-1]
            mplot.update_plot(y,x,offset=2,color=color,zoom=0,scale=scale) 
    
    return(pos_distro)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Play back images from a given directory')
    parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
    parser.add_argument('--dynamic',default  = 1 ,type = int)
    parser.add_argument('--dataset',
                                    default = 'kitti',
                                    type=str,
                                    help=' dataset root directory .'
                                    )
    parser.add_argument('--seq',    
                                default  = [00],
                                type = str)
    parser.add_argument('--plot',default  = True ,type = bool)
    parser.add_argument('--loop_thresh',default  = 1 ,type = float)
    parser.add_argument('--record_gif',default  = False ,type = bool)
    parser.add_argument('--option',default  = 'compt' ,type = str,choices=['viz','compt'])
    parser.add_argument('--pose_file',default  = 'poses' ,type = str)
    
    args = parser.parse_args()

    root    = args.root
    dataset = args.dataset 
    seq     = args.seq
    plotting_flag = args.plot
    record_gif_flag = args.record_gif
    option = args.option
    loop_thresh = args.loop_thresh

    print("[INF] Dataset Name:  " + dataset)
    print("[INF] Sequence Name: " + str(seq) )
    print("[INF] Plotting Flag: " + str(plotting_flag))
    print("[INF] record gif Flag: " + str(record_gif_flag))
    print("[INF] Reading poses from : " + args.pose_file)

    ground_truth = {'pos_range':25, # Loop Threshold [m]
                    'neg_range': 17,
                    'num_neg':10,
                    'num_pos':50,
                    'warmupitrs': 600, # Number of frames to ignore at the beguinning
                    'roi':500}
    
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions', 'kitti.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    loader = load_dataset('kitti',SESSION,memory='Disk')

    val = loader.get_val_loader()
    #dataset = dataset_loader(root,'kitti',seq,sync = True,ground_truth=ground_truth)
    
    table = val.dataset._get_gt_()
    true_loop = np.array([np.where(line==1)[0] for line in table])
    
    xy = val.dataset._get_pose_()
    n_point = xy.shape[0]
    yx = xy.copy()
    yx[0] = xy[1]
    yx[1] = xy[0]
    viz_overlap(yx,true_loop,record_gif=False)

 


    

