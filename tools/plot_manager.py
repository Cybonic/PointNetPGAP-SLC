import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tqdm


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



class Plot3DManager:
    def __init__(self, dataset, val_set, root2save = 'plots' ,
                 show_plot=True, save_plot_settings=True, topk=5, ground_truth=False, 
                 double_check= False, query_plot_factor = 3,
                 loop_range=10,window=10):
        """
        Initializes the PlotManager with the necessary parameters for loading and saving plot settings.
        
        Args:
            dataset (str): Dataset name.
            val_set (str): Validation set name.
            save_dir (str): Directory to save the plot and settings.
            show_plot (bool): Whether to display the plot.
            save_plot_settings (bool): Whether to save the plot settings.
            topk (int): The top-k predictions to plot.
            ground_truth (bool): Ground truth information for plotting.
        """
        self.dataset = dataset
        self.val_set = val_set
        self.show_plot = show_plot
        self.save_plot_settings = save_plot_settings
        self.topk = topk
        self.ground_truth = ground_truth
        self.root2save = root2save
        self.double_check = double_check
        self.query_plot_factor = query_plot_factor
        self.loop_range = loop_range
        self.window = window
        
        
        os.makedirs(self.root2save,exist_ok=True)
        # Create a directory to save the plot settings
        self.plot_setting_dir = os.path.join(root2save, "settings")
        os.makedirs(self.plot_setting_dir, exist_ok=True)

        # File name for the plot settings
        self.plot_setting_file = os.path.join(self.plot_setting_dir, f'{dataset.lower()}_{val_set.lower()}_matplotlibrc.json')
        self.view_settings = None
        
    


    def load_settings(self,ax):
        """
        Loads saved plot settings and applies them to the axis if available.
        
        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to apply the settings to.
            
        Returns:
            bool: Whether the settings were successfully loaded.
        """
        set_equal_axis = True
        if os.path.exists(self.plot_setting_file):
            with open(self.plot_setting_file, 'rb') as f:
                self.view_settings = pickle.load(f)
            
            print("Loading settings from:", self.plot_setting_file)
            ax.view_init(elev=self.view_settings['elev'], azim=self.view_settings['azim'])
            ax.set_xlim(self.view_settings['xlim'][0], self.view_settings['xlim'][1])
            ax.set_ylim(self.view_settings['ylim'][0], self.view_settings['ylim'][1])
            ax.set_zlim(self.view_settings['zlim'][0], self.view_settings['zlim'][1])
            set_equal_axis = False
        
        return set_equal_axis



    def save_settings(self, ax):
        """
        Saves the current plot settings to a file.
        
        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis to capture the current settings from.
        """
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
        
        if self.save_plot_settings:
            with open(self.plot_setting_file, 'wb') as f:
                pickle.dump(view_settings, f)
            print("Settings saved to:", self.plot_setting_file)

    
    def plot_place_2D_path(self,poses,**argv):
    #def plot_place_on_3D_map(self,poses,predictions,loop_range=1,topk=1,**argv):
        
        
        
        # Configuration parameters
        
        plot_path_point_scale = argv['plot_path_point_scale']  if 'plot_path_point_scale' in argv else 5
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        # Load and apply saved settings if available
        set_equal_axis = self.load_settings(self.ax)
        
        x = poses[:,0]
        y = poses[:,1]
        z = poses[:,2]
        
        x_subset = x
        y_subset = y
        
        
        z_scores = np.zeros_like(y_subset)
        
        self.ax.scatter(x_subset, y_subset, z_scores, color='k',s=plot_path_point_scale)
        
        # Set equal axis if the settings were not loaded
        if set_equal_axis:
            self.ax.axis('equal')
        
        # Turn off the axis
        self.ax.set_axis_off()
        
        # Show plot if enabled
        if self.show_plot:
            plt.show()
            # Save the plot settings
            self.save_settings(self.ax)
        
        

    def plot_place_on_3D_map(self,poses,predictions,loop_range=1,topk=1,**argv):
        
        
        # Configuration parameters
        vertical_scale = argv['scale'] if 'scale' in argv else 1
        segment_labels = argv['segment_labels']
        window = argv['window'] if 'window' in argv else 50
        plot_path_point_scale = argv['plot_path_point_scale']  if 'plot_path_point_scale' in argv else 5
        query_plot_factor = argv['query_plot_factor']  if 'query_plot_factor' in argv else 5
        double_check = argv['double_check'] if 'double_check' in argv else False
        
        x = poses[:,0]
        y = poses[:,1]
        z = poses[:,2]
        

        loops = comp_loop_overlap(poses,segment_labels,window,loop_range)
        
        x_subset = x
        y_subset = y
        
        
        z_subset = np.zeros_like(y_subset)
        # Compute z coordinate
        z_scores = np.linspace(0,10,len(np.unique(loops))) #vertical_scale*indices * (z[-1] - z[0]) / (len(x) - 1) + z[0]
        
        unique_loops = np.unique(loops)
        
        for score,lable in zip(z_scores,unique_loops):
            idx = np.where(loops==lable)[0]
            z_subset[idx] = score
        
        self.ax.scatter(x_subset, y_subset, z_subset, color='k',s=plot_path_point_scale)
        
        queries = np.array(list(predictions.keys()))
        
        # Plot a subset of points, equally distributed
        num_query_points = int(len(queries)/query_plot_factor)  # Number of query points to be  ploted
        plot_query_idx =  np.linspace(0,len(queries)-1,num_query_points,dtype=np.int32)
        
        query_list = queries[plot_query_idx]
        
        for itr,query in tqdm.tqdm(enumerate(query_list),total = len(query_list)):
        
            query_label = predictions[query]['segment']
            true_loops  = predictions[query]['true_loops']
            pred_loops  = predictions[query]['pred_loops']
            
            
            if self.ground_truth == False:
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
            
            pred_idx = pred_idx[pred_bool]
            
             # Double check if the distance is within the range
            if double_check == True:
                dist = np.linalg.norm(poses[query,:2] - poses[pred_idx,:2],axis=1)
                
                in_range_bool = dist < loop_range
                
                plot_idx = np.argsort(dist)[in_range_bool]
                
                if len(plot_idx) == 0:
                    continue
            else:
                plot_idx = np.arange(len(pred_idx),dtype=np.int32)
                
            plot_idx = plot_idx[0]
            

            plt.plot([x_subset[query],x_subset[pred_idx[plot_idx]]],
                     [y_subset[query],y_subset[pred_idx[plot_idx]]],
                     [z_subset[query],z_subset[pred_idx[plot_idx]]],color)

    
     
    def plot(self,xy, predictions, segment_labels,**argv):
        """
        Plots the 3D map and applies settings if available.
        
        Args:
            ax (matplotlib.axes._subplots.Axes3DSubplot): The axis to plot on.
            xy (np.ndarray): The xy positions for plotting.
            predictions (np.ndarray): Prediction data for plotting.
            segment_labels (np.ndarray): Labels for the segments.
            
        """
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Load and apply saved settings if available
        set_equal_axis = self.load_settings(self.ax)
        
        
        # Plot the 3D map
        self.plot_place_on_3D_map(xy, predictions, topk=self.topk, 
                                    loop_range=self.loop_range, 
                                    segment_labels = segment_labels, 
                                    ground_truth = self.ground_truth,
                                    double_check = self.double_check,
                                    query_plot_factor = self.query_plot_factor,
                                    window = self.window
                                    )
        
        
        
        # Set equal axis if the settings were not loaded
        if set_equal_axis:
            self.ax.axis('equal')
        
        # Turn off the axis
        self.ax.set_axis_off()
        
        # Show plot if enabled
        if self.show_plot:
            plt.show()
            # Save the plot settings
            self.save_settings(self.ax)

        
        
        
    def save_fig(self,file_name):
        # Save the plot
        plot_file_path = f'{file_name}.pdf'
        plt.savefig(plot_file_path, transparent=True, bbox_inches='tight')
        
        plot_file_path = f'{file_name}.png'
        plt.savefig(plot_file_path, transparent=True, bbox_inches='tight')
        print("*" * 50)
        print("Plot saved to:", plot_file_path)
        print("*" * 50)

        
        



