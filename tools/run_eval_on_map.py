#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


# Getting latend space using Hooks :
#  https://towardsdatascience.com/the-one-pytorch-trick-which-you-should-know-2d5e9c1da2ca

# Binary Classification
# https://jbencook.com/cross-entropy-loss-in-pytorch/


import argparse
import yaml
import os
import torch 
import sys
import tqdm
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

#from networks.orchnet import *
from trainer import Trainer
from pipeline_factory import model_handler,dataloader_handler
import numpy as np
from utils.viz import myplot

from plotting_settings import SETTINGS

import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

def find_file(target_file, search_path):
    assert os.path.exists(search_path), "The search path does not exist: " + search_path
    for root, dirs, files in os.walk(search_path):
        if target_file in files:
            return os.path.join(root, target_file)
    return ''


def plot_sim_on_3D_map(poses,predictions,samples = 10000,sim_thresh=0.5,loop_range=1,topk=1,**argv):
    import matplotlib.pyplot as plt

    vertical_scale = argv['scale'] if 'scale' in argv else 1
    descriptors = argv['descriptors']
    
    np_descriptors = np.array([descriptor['d'] for descriptor in descriptors.values()])
    
    x = poses[:,0]
    y = poses[:,1]
    z = poses[:,2]
    
    # Plot a subset of points, equally distributed
    num_points = poses.shape[0]  # Number of points to plot
    indices = np.linspace(0, len(x)-1, num_points, dtype=int)  # Equally spaced indices

    
    x_subset = x#[indices]
    y_subset = y#[indices]
    z_subset = vertical_scale*indices * (z[-1] - z[0]) / (len(x) - 1) + z[0]
    
    # Create a 2D path plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Colorize points based on label
    colors = plt.cm.get_cmap('viridis', 7)
    
    queries = np.array(list(predictions.keys()))
    # select randomly 1 query out of all queries
    
    plot_idx = np.random.randint(0,len(queries),1)

    plot_idx = [2000]
    query = queries[plot_idx][0]
    print(f'Plotting query {query}')
    
    query_descriptor = descriptors[query]['d']
        
    sim = np.linalg.norm(query_descriptor - np_descriptors,axis=1)
    
    # Rescale exp the similarity
    sigma = 0.3
    sim = 1-np.exp(-sim/sigma)
    # Colorize points based on similarity
    ax.scatter(x_subset, y_subset, z_subset, c=sim, cmap='YlGn',s=5)
    ax.scatter(x_subset[query], y_subset[query], z_subset[query], c='k',s=30)
        
    ax.axis('equal')  # Equal aspect ratio
    ax.set_axis_off()  # Turn off the axis
    plt.show()
    
    for key, value in plt.rcParams.items():
        print(f"{key}: {value}")


def comp_loop_overlap(positions:np.ndarray,segment_labels:np.ndarray,window:int,rth:float):
    """Computes the loops 

    Args:
        pose (np.ndarray): nx3 array of positions
        segment_labels (np.ndarray): nx1 array of segment labels
        window (_type_): _description_
        rth (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    
    #from sklearn.metrics import pairwise_distances
    from scipy.spatial.distance import cdist
    
    #pair_L2_dis = cdist(positions,positions)
    
    lower_bound = window+10
    
    labels = np.zeros(positions.shape[0])
    for i,p in enumerate(positions):
        
        if i < lower_bound:
            continue
        
        query_segment_label = segment_labels[i]
        
        upper_bound = i-window
        map_idx = np.linspace(1,upper_bound,upper_bound,dtype=np.int32)
        
        p = positions[i].reshape(1,-1)
        
        # Compute L2 distance
        distances = np.linalg.norm(p-positions[map_idx,:],axis=1)
        
        # Get target segment labels
        map_segment_labels = segment_labels[map_idx]
        
        # 
        i_sort = np.argsort(distances)
        loop_bool = distances[i_sort[0]]<rth 
        
        segment_match = query_segment_label == map_segment_labels[i_sort[0]]
        labels[i]=labels[i-1] # this maintains the path flat 
        if np.sum(loop_bool)>0 and np.sum(segment_match):
            labels[i]= labels[i-1] +1  # increment label, this makes the path increase in zzz 
    
    return labels
    
    
    


def plot_place_on_3D_map(poses,predictions,loop_range=1,topk=1,**argv):
    

    vertical_scale = argv['scale'] if 'scale' in argv else 1
    segment_labels = argv['segment_labels']
    ground_truth = argv['ground_truth'] if 'ground_truth' in argv else False
    
    x = poses[:,0]
    y = poses[:,1]
    z = poses[:,2]
    
    # Plot a subset of points, equally distributed
    num_points = int(poses.shape[0])  # Number of points to plot
    indices = np.linspace(0, len(x)-1, num_points, dtype=int)  # Equally spaced indices

    lebels = comp_loop_overlap(poses,segment_labels,50,loop_range)
    
    x_subset = x#[indices]
    y_subset = y#[indices]
    
    
    z_subset = np.zeros_like(y_subset)
    # Compute z coordinate
    z_scores = np.linspace(0,10,len(np.unique(lebels))) #vertical_scale*indices * (z[-1] - z[0]) / (len(x) - 1) + z[0]
    
    unique_lables = np.unique(lebels)
    
    for score,lable in zip(z_scores,unique_lables):
        idx = np.where(lebels==lable)[0]
        z_subset[idx] = score
    
    ax.scatter(x_subset, y_subset, z_subset, color='k',s=5)
    

    queries = np.array(list(predictions.keys()))
    
    # Plot a subset of points, equally distributed
    num_query_points = int(len(queries)/3)  # Number of points to plot
    #indices = np.linspace(0, len(x)-1, num_query_points, dtype=int)  # Equally spaced indices
    plot_query_idx =  np.linspace(0,len(queries)-1,num_query_points,dtype=np.int32)
    
    query_list = queries[plot_query_idx]
    
    for itr,query in tqdm.tqdm(enumerate(query_list),total = len(query_list)):
        
        c = np.array(['k']*(query+1)) # set to gray by default
        #c[:query] = ['k']*query
        s = np.ones(query+1)*10
        
    
        query_label = predictions[query]['segment']
        true_loops  = predictions[query]['true_loops']
        pred_loops = predictions[query]['pred_loops']
        
        max_values = max(true_loops['dist'] )
        
        if ground_truth == False:
            pred_idx = pred_loops['idx'][:topk]
            pred_label = pred_loops['segment'][:topk]
        else:
            pred_idx   = true_loops['idx'][:topk]
            pred_label = true_loops['segment'][:topk]


        pred_bool =  (query_label == pred_label) # * (pred_dist < loop_range)
        
        if (pred_bool).any():
            color = 'g'
        else:
            continue
            #color = 'r'
        
        pred_idx = pred_idx[pred_bool]
        
        # Double check if the distance is within the range
        dist = np.linalg.norm(poses[query,:2] - poses[pred_idx,:2],axis=1)
        
        in_range_bool = dist < loop_range
        
        plot_idx = np.argsort(dist)[in_range_bool]
        
        if len(plot_idx) == 0:
            continue
        
        plot_idx = plot_idx[0]
        

        plt.plot([x_subset[query],x_subset[pred_idx[plot_idx]]],
                [y_subset[query],y_subset[pred_idx[plot_idx]]],
                [z_subset[query],z_subset[pred_idx[plot_idx]]],color)

    
    



 

def plot_retrieval_on_map(poses,predictions,sim_thresh=0.5,loop_range=1,topk=1,record_gif=False,**argv):
    # Save Similarity map

    save_dir =''
    if 'save_dir' in argv:
        save_dir = argv['save_dir']
        
    file_name = f'experiment'
    if 'gif_name' in argv:
            file_name = argv['gif_name']
    
    save_steps_flag = False
    save_step_dir = ''
    save_step_itrs = []
    
    if 'save_step_itr' in argv and  isinstance(argv['save_step_itr'],list):
        save_step_itrs = argv['save_step_itr']
        save_step_dir = os.path.join(save_dir,file_name)
        os.makedirs(save_step_dir,exist_ok=True)
    
    
    plot = myplot(delay = 0.001)
    plot.init_plot(poses[:,0],poses[:,1],c='k',s=10)
    plot.xlabel('m')
    plot.ylabel('m')
                     
    if record_gif == True:
        # Build the name of the file
        name = os.path.join(save_dir,file_name+'.gif') # Only add .gif here because the name is used below as name of a dir
        plot.record_gif(name)

    keys = list(predictions.keys())
    first_point = keys[0].item()
    
    n_samples = poses.shape[0]
    
    true_positive = []
    wrong = []
    query_list = list(range(first_point,n_samples,20))
    for itr,query in tqdm.tqdm(enumerate(query_list),total = len(query_list)):
        
        c = np.array(['k']*(query+1)) # set to gray by default
        #c[:query] = ['k']*query
        s = np.ones(query+1)*10
        
    
        query_label = predictions[query]['segment']
        true_loops = predictions[query]['true_loops']
        cand_loops = predictions[query]['pred_loops']
        
        # array of booleans
        knn = np.zeros(len(cand_loops['idx']),dtype=bool)
        knn[:topk] = True
        
        positives = cand_loops['sim'] < sim_thresh 
        inrange   = cand_loops['dist'] < loop_range
        inrow     = cand_loops['labels'] == query_label
        
        bool_cand = positives*inrange*inrow*knn
        
        cand_idx = cand_loops['idx'][bool_cand]
        loop_idx = true_loops['idx'][true_loops['dist'] < loop_range][:topk]

        c[query] = 'b'
        s[query] = 80
        
        
                
        if len(loop_idx) > 0 and len(cand_idx) > 0:
            true_positive.extend(cand_idx)
            
        if len(loop_idx) > 0 and len(cand_idx) == 0:
            wrong.append(query)
           
        if len(loop_idx) == 0 and len(cand_idx) > 0:
            wrong.append(query)
        
        np_true_positive = np.array(true_positive,dtype=np.int32).flatten()
        np_wrong = np.array(wrong,dtype=np.int32).flatten()
        
        c[np_true_positive] = 'g'
        s[np_true_positive] = 150
        
        c[np_wrong] = 'r'
        s[np_wrong] = 50

        plot.update_plot(poses[:query+1,0],poses[:query+1,1],color = c , offset= 1, zoom=-1,scale=s)
        
         # save png of parts of the plot
        if itr in save_step_itrs:
            plot.save_plot(os.path.join(save_step_dir,f'{itr}.png'))



def plot_place_on_map(poses,predictions,topk=1,record_gif=False,loop_range=10,**argv):
    # Save Similarity map

    save_dir =''
    if 'save_dir' in argv:
        save_dir = argv['save_dir']
        
    file_name = f'experiment'
    if 'gif_name' in argv:
            file_name = argv['gif_name']
    
    
    save_steps_flag = False
    save_step_dir = ''
    save_step_itrs = []
    
    if 'save_step_itr' in argv and  isinstance(argv['save_step_itr'],list):
        save_step_itrs = argv['save_step_itr']
        save_step_dir = os.path.join(save_dir,file_name)
        os.makedirs(save_step_dir,exist_ok=True)
    
    
    plot = myplot(delay = 0.001)
    plot.init_plot(poses[:,0],poses[:,1],c='k',s=10)
    plot.xlabel('m')
    plot.ylabel('m')
                     
    if record_gif == True:
        # Build the name of the file
        name = os.path.join(save_dir,file_name+'.gif') # Only add .gif here because the name is used below as name of a dir
        plot.record_gif(name)
    
    queries = np.array(list(predictions.keys()))
    
    plot_query_idx = list(range(0,len(queries),20))
    
    n_samples = poses.shape[0]
    
    true_positive = []
    wrong = []
    query_list = queries[plot_query_idx]
    for itr,query in tqdm.tqdm(enumerate(query_list),total = len(query_list)):
        
        c = np.array(['k']*(query+1)) # set to gray by default
        #c[:query] = ['k']*query
        s = np.ones(query+1)*10
        
    
        query_label = predictions[query]['segment']
        true_loops = predictions[query]['true_loops']
        pred_loops = predictions[query]['pred_loops']
        
        max_values = max(true_loops['dist'] )
        
        pred_idx = pred_loops['idx'][:topk]
        pred_label = pred_loops['segment'][:topk]
        #pred_dist = pred_loops['dist'][:topk]
        
        pred_bool = pred_label == query_label #and pred_dist < loop_range
        
        #loop_idx = true_loops['idx'][true_loops['dist'] < loop_range][:topk]

        c[query] = 'b'
        s[query] = 80
        
        
                
        if (pred_bool).all():
            #true_positive.extend(pred_idx)
            true_positive = pred_idx
            wrong = []
        else:
            #wrong.append(query)
            true_positive = []
            wrong = query
                
        
        np_true_positive = np.array(true_positive,dtype=np.int32).flatten()
        np_wrong = np.array(wrong,dtype=np.int32).flatten()
        
        c[np_true_positive] = 'g'
        s[np_true_positive] = 150
        
        c[np_wrong] = 'r'
        s[np_wrong] = 50

        plot.update_plot(poses[:query+1,0],poses[:query+1,1],color = c , offset= 1, zoom=-1,scale=s)
        
         # save png of parts of the plot
        if itr in save_step_itrs:
            plot.save_plot(os.path.join(save_step_dir,f'{itr}.png'))
        
 
        
        
def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--dataset_root',type=str, required=False,
        default='/home/tiago/workspace/DATASET',
        help='Directory to the dataset root'
    )
    
    parser.add_argument(
        '--network', type=str,
        default='SPVSoAP3D', help='model to be used'
    )

    parser.add_argument(
        '--experiment',type=str,
        default='thesis/horto_predictions',
        help='Name of the experiment to be executed'
    )

    parser.add_argument(
        '--memory', type=str,
        default='DISK',
        choices=['DISK','RAM'],
        help='RAM: loads the dataset to the RAM memory first. DISK: loads the dataset on the fly from the disk'
    )

    parser.add_argument(
        '--device', type=str,
        default='cuda',
        help='Directory to get  the trained model.'
    )
    parser.add_argument(
        '--batch_size',type=int,
        default=10,
        help='Batch size'
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        required=False,
        default = "eval/ground_truth_loop_range_10m.pkl",
        help='sampling points.'
    )

    parser.add_argument(
        '--loop_range',
        type=float,
        required=False,
        default = 10,
        help='loop range to monitor the performance.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='HORTO-3DLM', # uk
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default = 'SJ23',
        help = 'Validation set'
    )

    parser.add_argument(
        '--roi',
        type=float,
        required=False,
        default = 0,
        help = 'Crop range [m] to crop the point cloud around the scan origin.'
    )
    
    parser.add_argument(
        '--resume', '-r',
        type=str,
        required=False,
        default='/home/tiago/workspace/pointnetgap-RAL',
        # #LOGG3D-LazyTripletLoss_L2-segment_lossM0.1-descriptors
        # #PointNetVLAD-LazyTripletLoss_L2-segment_loss-m0.5'
        # #overlap_transformer-LazyTripletLoss_L2-segment_loss-m0.5
        help='Directory to get the trained model or descriptors.'
    )

    parser.add_argument(
        '--session',
        type=str,
        required=False,
        default = "ukfrpt",
    )
    
    parser.add_argument(
        '--eval_roi_window',
        type=float,
        required=False,
        default = 300,
        help='Number of frames to ignore in imidaite vicinity of the query frame.'
    )
    
    parser.add_argument(
        '--eval_warmup_window',
        type=float,
        required=False,
        default = 100,
        help='Number of frames to ignore in the beginning of the sequence'
    )
    
    parser.add_argument(
        '--eval_protocol',
        type=str,
        required=False,
        choices=['place'],
        default = 'place',
    )
    
    parser.add_argument(
        '--plot_on_map',
        type=str,
        required=False,
        default = 'saved_model_data',
    )
    parser.add_argument(
        '--topk',
        type=int,
        required=False,
        default = 1,
    )
    parser.add_argument(
        '--ground_truth',
        type=bool,
        required= False,
        default = True,
    )
    parser.add_argument(
        '--show_plot',
        type=bool,
        required=False,
        default = False,
    )
    
    parser.add_argument(
        '--rerun_flag',type=bool,
        default = False,
        help='sampling points.'
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    # Update config file with new settings
    SESSION['experiment'] = FLAGS.experiment
    
    # Define evaluation mode: cross_validation or split
    SESSION['train_loader']['triplet_file'] = None
    
    # Update the validation loader
    SESSION['val_loader']['batch_size'] = FLAGS.batch_size
    SESSION['val_loader']['ground_truth_file'] = FLAGS.eval_file
    SESSION['val_loader']['augmentation'] = False
    
    # Update the model settings
    SESSION['roi'] = FLAGS.roi
    SESSION['rerun_flag'] = FLAGS.rerun_flag
    SESSION['memory']     = FLAGS.memory
    SESSION['loop_range']   = FLAGS.loop_range
    SESSION['eval_roi_window'] = FLAGS.eval_roi_window
    SESSION['descriptor_size'] = 256
    SESSION['eval_warmup_window'] = FLAGS.eval_warmup_window
    SESSION['eval_protocol'] = FLAGS.eval_protocol
    SESSION['device'] = FLAGS.device


    print("----------")
    print("\n======= VAL LOADER =======")
    print("Batch Size : ", str(SESSION['val_loader']['batch_size']))
    print("Max Points: " + str(SESSION['max_points']))
    print("\n========== MODEL =========")
    print("Backbone : ", FLAGS.network)
    print("Resume: ",  FLAGS.resume )
    #print("MiniBatch Size: ", str(SESSION['modelwrapper']['minibatch_size']))
    print("\n==========================")
    print(f'Eval Protocal: {FLAGS.eval_protocol}')
    print(f'Memory: {FLAGS.memory}')
    print(f'Device: {FLAGS.device}')
    print("Experiment: %s" %(FLAGS.experiment))
    print("----------\n")

    # For repeatability
    
    torch.manual_seed(0)
    np.random.seed(0)

    if os.path.isfile(FLAGS.resume):
        print("Resuming form %s"%FLAGS.resume)
        
        resume_struct= FLAGS.resume.split('/')
        assert FLAGS.val_set in resume_struct, "The resume file does not match the validation set"
        assert FLAGS.network in resume_struct, "The resume file does not match the network" 
    
    ###################################################################### 
    loader = dataloader_handler(FLAGS.dataset_root,
                                FLAGS.network,
                                FLAGS.dataset,
                                FLAGS.val_set,
                                SESSION, 
                                roi = FLAGS.roi, 
                                pcl_norm = False,
                                model_evaluation='cross_evaluation')
     
    from place_recognition import PlaceRecognition 

    #loop_range = FLAGS.loop_range
    
    import logging
    log_file    = os.path.join('logs',f'{FLAGS.experiment}.log')
    logger      = logging.getLogger(__name__)
    #log_handler = logging.FileHandler(log_file)
    
    # LOAD PLACE RECOGNITION
    eval_approach = PlaceRecognition(
        None,
        loader.get_val_loader(),
        top_cand = 1,
        logger   = logger,
        roi_window   = FLAGS.eval_roi_window,
        warmup_window = FLAGS.eval_warmup_window,
        device  = FLAGS.device,
        logdir =  FLAGS.experiment,
        monitor_range = FLAGS.loop_range,
        sim_func = 'L2',
        eval_protocol = FLAGS.eval_protocol
        )
    
    resume = FLAGS.resume
    predictions = os.path.join(resume,FLAGS.experiment,f'#{FLAGS.network}',f'eval-{FLAGS.val_set}')
    
    print("*"*50)
    print("\nLoading predictions from: ",predictions)
    print("*"*50)
    
    prediction_file = find_file('predictions.pkl',predictions)
    descriptors_file = find_file('descriptors.torch',predictions)
    eval_approach.load_descriptors(descriptors_file)
    
    predictions = []
    # if the prediction file does not exist, generate the predictions
    if not os.path.exists(prediction_file) or FLAGS.rerun_flag:
        # Generate descriptors and predictions  
        eval_approach.run(loop_range = FLAGS.loop_range)
        # Save the predictions
        predictions = eval_approach.save_predictions_pkl(save_dir='temp')
        
    else:
        predictions = eval_approach.load_predictions_pkl(prediction_file)
    
    # Load the predictions and descriptors
    
    descriptors = eval_approach.descriptors
    
    
    # Load the poses and segment labels
    poses = eval_approach.poses
    segment_labels = eval_approach.row_labels
    sequence = FLAGS.val_set
    
    # Load aligned rotation
    xy = poses
    
    from tools.plot_manager import Plot3DManager
    
    
    plot = Plot3DManager(FLAGS.dataset, FLAGS.val_set,
                         loop_range   = FLAGS.loop_range, # loop range
                         ground_truth = FLAGS.ground_truth, # flag 
                         topk = FLAGS.topk,
                         show_plot = FLAGS.show_plot,
                         window=FLAGS.eval_roi_window,
                         query_plot_factor=1)
    
    plot.plot(poses,predictions, segment_labels=segment_labels)
    # Save gif
    
     # File name for the plot file
    plot_dir = os.path.join('plots', "figs")
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_file = ''
    if FLAGS.ground_truth == False:
        plot_file = os.path.join(plot_dir, f'3d_plot_{FLAGS.dataset.lower()}_{FLAGS.val_set.lower()}_{FLAGS.loop_range}_{FLAGS.topk}')
    else:
        plot_file = os.path.join(plot_dir, f'3d_plot_ground_truth_{FLAGS.dataset.lower()}_{FLAGS.val_set.lower()}_{FLAGS.loop_range}_{FLAGS.topk}')
    
    plot.save_fig(plot_file)
    
    if FLAGS.ground_truth:
        # PLOT 2D PATH
        plot.plot_place_2D_path(poses)
        plot_file = os.path.join(plot_dir, f'2D_plot_ground_truth_{FLAGS.dataset.lower()}_{FLAGS.val_set.lower()}')
        plot.save_fig(plot_file)
    
    exit()
    
    
    
    model_name = FLAGS.network.lower() if FLAGS.ground_truth == False else "GroundTruth"
    file_name = '-'.join([FLAGS.dataset.lower(),FLAGS.val_set.lower(),model_name,f'topk{FLAGS.topk}'])
 
    root2save = os.path.join('plots','retrieval_on_map',f'range{loop_range}m')
    os.makedirs(root2save,exist_ok=True)
    print("Saving to: ",root2save)
    
    #file =  
    
    save_itrs = list(range(1,len(predictions.keys()),10))
    
    
    
    #plot_place_on_map(xy, predictions,topk = FLAGS.topk, record_gif = True, gif_name = file_name,save_dir = root2save, 
    #                 save_step_itr = save_itrs,loop_range = loop_range)
    # Load the dictionary from the JSON file
    

    
    
    
