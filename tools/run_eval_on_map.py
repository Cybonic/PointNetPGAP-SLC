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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

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
    

def plot_place_on_3D_map(poses,predictions,samples = 10000,sim_thresh=0.5,loop_range=1,topk=1,**argv):
    

    vertical_scale = argv['scale'] if 'scale' in argv else 1
    
    segment_labels = argv['segment_labels']
    
    ground_truth = argv['ground_truth'] if 'ground_truth' in argv else False
    
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
    

    # Colorize points based on label
    colors = plt.cm.get_cmap('viridis', 7)
    #colors = plt.cm.get_cmap('viridis', len(unique_labels))
    ax.scatter(x_subset, y_subset, z_subset, color='k',s=5)
    
    queries = np.array(list(predictions.keys()))
    
    plot_query_idx = list(range(0,len(queries),1))
    
    n_samples = poses.shape[0]
    
    true_positive = []
    wrong = []
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
        #pred_dist = pred_loops['dist'][:topk]
        
        # True Positive 
        query_label_gt = segment_labels[query]
        pred_labels_gt = segment_labels[pred_idx]
        
        #assert (query_label == query_label_gt).all(), "The query label is not the same as the ground truth"
        
        pred_bool =  (query_label == pred_label) # * (pred_dist < loop_range)
        # pred_bool_dis = pred_dist < loop_range
        
        
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
        
        
        #if (in_range_bool).any():
        #    color = 'b'
            #continue
        
        
        
        plt.plot([x_subset[query],x_subset[pred_idx[plot_idx]]],
                 [y_subset[query],y_subset[pred_idx[plot_idx]]],
                 [z_subset[query],z_subset[pred_idx[plot_idx]]],color)
        #loop_idx = true_loops['idx'][true_loops['dist'] < loop_range][:topk]

        #c[query] = 'b'
        #s[query] = 80
        
        
        
   
    
    



 

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
        default='PointNetVLAD', help='model to be used'
    )

    parser.add_argument(
        '--experiment',type=str,
        default='kitti_test_predictions',
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
        '--max_points',type=int,
        default = 10000,
        help='sampling points.'
    )

    parser.add_argument(
        '--eval_file',
        type=str,
        required=False,
        default = "eval/ground_truth_loop_range_10m.pkl",
        help='sampling points.'
    )

    parser.add_argument(
        '--monitor_loop_range',
        type=float,
        required=False,
        default = 10,
        help='loop range to monitor the performance.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='kitti', # uk
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--val_set',
        type=str,
        required=False,
        default = '06',
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
        default='/home/tiago/workspace/pointnetgap-RAL/RALv3',
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
        '--save_predictions',
        type=str,
        required=False,
        default = 'saved_model_data',
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
        default = False,
    )
    parser.add_argument(
        '--show_plot',
        type=bool,
        required=False,
        default = False,
    )
    parser.add_argument(
        '--save_plot_settings',
        type=bool,
        required=False,
        default = False,
    )
    
    FLAGS, unparsed = parser.parse_known_args()

    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    session_cfg_file = os.path.join('sessions', FLAGS.session + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    SESSION['save_predictions'] = FLAGS.save_predictions
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
    SESSION['max_points'] = FLAGS.max_points
    SESSION['memory']     = FLAGS.memory
    SESSION['monitor_range']   = FLAGS.monitor_loop_range
    SESSION['eval_roi_window'] = FLAGS.eval_roi_window
    SESSION['descriptor_size'] = 256
    SESSION['eval_warmup_window'] = FLAGS.eval_warmup_window
    SESSION['eval_protocol'] = FLAGS.eval_protocol
    SESSION['device'] = FLAGS.device


    print("----------")
    print("Saving Predictions: %s"%FLAGS.save_predictions)
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
                                model_evaluation='cross_domain')
     
    from place_recognition import PlaceRecognition 

    loop_range = FLAGS.monitor_loop_range
    
    import logging
    log_file    = os.path.join('logs',f'{FLAGS.experiment}.log')
    logger      = logging.getLogger(__name__)
    log_handler = logging.FileHandler(log_file)
    
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
        monitor_range = loop_range,
        sim_func='L2',
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
    
    # if the prediction file does not exist, generate the predictions
    if not os.path.exists(prediction_file):
        # Generate descriptors and predictions  
        eval_approach.run(loop_range = loop_range)
        # Save the predictions
        prediction_file = eval_approach.save_predictions_pkl()
        # Save the parameters
        eval_approach.save_params()
        # Save the results
        eval_approach.save_results_csv()
    
    # Load the predictions and descriptors
    descriptors = eval_approach.descriptors
    predictions = eval_approach.load_predictions_pkl(prediction_file)
    
    # Load the poses and segment labels
    poses = eval_approach.poses
    segment_labels = eval_approach.row_labels
    sequence = FLAGS.val_set
    
    # Load aligned rotation
    xy = poses

    
    
    # Save gif
    
    model_name = FLAGS.network.lower() if FLAGS.ground_truth == False else "GroundTruth"
    file_name = '-'.join([FLAGS.dataset.lower(),FLAGS.val_set.lower(),model_name,f'topk{FLAGS.topk}'])
 
    root2save = os.path.join('plots','retrieval_on_map',f'range{loop_range}m')
    os.makedirs(root2save,exist_ok=True)
    print("Saving to: ",root2save)
    
    #file =  
    
    save_itrs = list(range(1,len(predictions.keys()),10))
    
    #plot_place_on_map(xy, predictions,topk = FLAGS.topk, record_gif = True, gif_name = file_name,save_dir = root2save, 
    #                  save_step_itr = save_itrs,loop_range = loop_range)
    # Load the dictionary from the JSON file
    
    import json
    import pickle
    
    # Create a directory to save the plot settings 
    plot_setting_dir = os.path.join("plots","settings")
    os.makedirs(plot_setting_dir,exist_ok=True)
    
    
    # File name for the plot settings
    plot_setting_file = os.path.join(plot_setting_dir,f'{FLAGS.dataset.lower()}_{FLAGS.val_set.lower()}_matplotlibrc.json')
    
    
    set_equal_axis = True
    if os.path.exists(plot_setting_file):
        
        with open(plot_setting_file, 'rb') as f:
            view_settings = pickle.load(f)

        print("Loading settings from: ",plot_setting_file)
        # Apply the loaded settings
        import matplotlib.pyplot as plt
        ax.view_init(elev=view_settings['elev'], azim=view_settings['azim'])
        # Restore the axis limits to achieve the same "zoom"
        ax.set_xlim(view_settings['xlim'][0], view_settings['xlim'][1])
        ax.set_ylim(view_settings['ylim'][0], view_settings['ylim'][1])
        ax.set_zlim(view_settings['zlim'][0], view_settings['zlim'][1])
        set_equal_axis = False
        #ax.dist = view_settings['camera_distance'] * 0.01
        # ax.dist = view_settings['dist']

    
    
    plot_place_on_3D_map(xy,predictions,topk = FLAGS.topk,record_gif = True,gif_name = file_name, save_dir = root2save,
                         save_step_itr = save_itrs,loop_range = loop_range,segment_labels = segment_labels,scale = SETTINGS[FLAGS.val_set]['scale'],
                         ground_truth = FLAGS.ground_truth
                          )

    # only set the axis if the settings are not loaded
    if set_equal_axis:
        ax.axis('equal')  # Equal aspect ratio
        
    ax.set_axis_off()  # Turn off the axis
    if FLAGS.show_plot:
        plt.show()
   
    # save current plot
    plt.savefig(os.path.join(root2save,f'{file_name}.png'),transparent=True,bbox_inches='tight')
    
    print("*"*50)
    print("Plot saved to: ",os.path.join(root2save,f'{file_name}.png'))
    print("*"*50)
    
    # Get the current view settings
    view_settings = {
            'elev': ax.elev,
            'azim': ax.azim,
            'xlim': ax.get_xlim(),
            'ylim': ax.get_ylim(),
            'zlim': ax.get_zlim(),
            'camera_distance': ax.get_proj()[0, 0]  # Approximation for camera distance
    }

    for key, value in view_settings.items():
        print(f"{key}: {value}")
    
    if FLAGS.save_plot_settings:
        with open(plot_setting_file, 'wb') as f:
            pickle.dump(view_settings, f)

    print("Settings saved to: ",plot_setting_file)
    
    
    
    
