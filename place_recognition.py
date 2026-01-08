#!/usr/bin/env python3

import argparse
import yaml
import os
os.environ['NUMEXPR_NUM_THREADS'] = '16'

import torch 
from tqdm import tqdm
import numpy as np
from networks import contrastive
from utils.eval import eval_row_place,eval_row_relocalization
import logging
import pandas as pd

from utils.utils import get_available_devices
from pipeline_factory import model_handler,dataloader_handler
import pickle

def compute_segment_pred(preds,targets):
    # compute the confusion matrix
    confusion_matrix = np.zeros((6,6))
    
    # compute confusion matrix
    for pred,gt in zip(preds,targets):
        confusion_matrix[gt,pred] += 1
        
    # compute accuracy
    accuracy = np.sum(targets == preds)/len(preds)
    
    # compute f1 score
    f1_score = np.zeros(6)
    for i in range(6):
        tp = confusion_matrix[i,i]
        fp = np.sum(confusion_matrix[i,:]) - tp
        fn = np.sum(confusion_matrix[:,i]) - tp
        f1_score[i] = 2*tp/(2*tp+fp+fn)
        
    
    restus = {'confusion_matrix':confusion_matrix,
              'accuracy':accuracy,
              'f1_score':f1_score}
    
    return restus
        
        

def search_files_in_dir(directory,search_file):
    files_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(search_file):
                files_found.append(os.path.join(root, file))
    return files_found


    
class PlaceRecognition:
    """
    Place Recognition evaluation and descriptor generation class.
    
    Attributes:
        model: Neural network model for descriptor generation
        loader: Data loader for retrieving samples
        device: Device to run the model on (cpu/cuda)
        logger: Logger instance for tracking progress
    """
    
    VALID_SIM_FUNCS = ['L2', 'cosine']
    VALID_EVAL_PROTOCOLS = ['place', 'relocalization']
    
    def __init__(self, model, loader, retrieval, logger, run_config, run_name, device, **kwargs):
        """
        Initialize PlaceRecognition evaluator.
        
        Args:
            model: Neural network model
            loader: Data loader
            top_cand: Top candidates for retrieval
            logger: Logger instance
            roi_window: Region of interest window size
            warmup_window: Warmup window size
            save_deptrs: Whether to save descriptors
            device: Device to use (cpu/cuda)
            eval_protocol: Evaluation protocol (place/relocalization)
            monitor_range: Monitoring range in meters
            sim_func: Similarity function (L2/cosine)
            save_predictions: Directory to save predictions
            **kwargs: Additional arguments (logdir)
        """
        self.sim_func = retrieval['sim_metric']
        task = run_config['task']
        self.run_name = run_name 
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        # Validate inputs
        assert self.sim_func in self.VALID_SIM_FUNCS, f'Invalid similarity function: {self.sim_func}'
        assert task in self.VALID_EVAL_PROTOCOLS, f'Invalid evaluation protocol: {task}'
        
        # Core attributes
        self.model = model
        self.loader = loader
        self.logger = logger or self._create_default_logger(model, task, sim_func)
        self.device = self._setup_device(device)
        
        # Configuration
        self.top_cand       = retrieval['top_cand']
        self.roi_window     = retrieval['roi_window']
        self.warmup_window  = retrieval['warmup_window']
        self.monitor_range  = run_config['range_report']
        self.sim_func       = self.sim_func
        self.task           = task
        self.save_deptrs    = run_config['save_deptrs']
        self.use_load_deptrs = False
        
        # Dataset and model info
        self.save_dir = os.path.join(self.root,
                                    run_config['save_dir'],
                                    self.run_name['experiment'],
                                    self.run_name['model'],
                                    self.run_name['seq']
                                   )
        os.makedirs(self.save_dir, exist_ok=True)
        print("Save directory:", self.save_dir)
        # Dataset attributes
        self._load_dataset_info(loader)
        
        # Setup prediction directory
        logdir = kwargs.get('logdir', 'default')
        self.predictions_dir = os.path.join(self.save_dir,'predictions')

        self.descripts_file = os.path.join(self.save_dir, 'descriptors.torch')
        # Parameters storage
        self.param = self._create_param_dict(self.top_cand, self.roi_window, self.warmup_window)

        # Log configuration
        self._log_configuration()
        
        
        
        
    def load_hortov2_data(self, dist_tresh=10):
        """Load Hortov2 dataset ground truth and positions."""
        # Try to load existing ground truth from file
        
        descripts_file = os.path.join(self.save_dir, 'descriptors.torch')
        
        if os.path.isfile(descripts_file):
            self.global_descriptors = self.load_descriptors(descripts_file)
        
        ground_truth = self.loader.dataset.load_ground_truth()
        
        # If no ground truth file exists, compute it
        if ground_truth is None:
            ground_truth = self.loader.dataset.get_ground_truth_loop_closure(
                warm_up=self.warmup_window,
                lower_bound_idx=self.roi_window,
                distance_threshold=dist_tresh,
                top_k=1
            )
        
        # Extract data from ground truth
        # Convert data structure to Place Recognition format
        self.anchors = ground_truth['query_indices']
        self.positions = self.loader.dataset.get_positions()
        self.row_labels = self.loader.dataset.get_labels()
        
        print("*" * 100)
        print("[INFO] Loaded Hortov2 Data")
        print(f"[INFO] anchors: {len(self.anchors)}")
        print(f"[INFO] positions: {len(self.positions)}")
        print(f"[INFO] labels: {len(self.row_labels)}")
        print("*" * 100)
        print(f"[INFO] positions: {len(self.positions)}")
        print(f"[INFO] labels: {len(self.row_labels)}")
        print("*"*100)
    
    def _setup_device(self, device):
        """Setup and validate device."""
        if device in ['gpu', 'cuda']:
            device, _ = get_available_devices(1, self.logger)
        self.logger.info(f'Using device: {device}')
        return device
    
    def _create_default_logger(self, model, eval_protocol, sim_func):
        """Create default logger if none provided."""
        log_file = os.path.join('logs', f'{model}_{eval_protocol}_{sim_func}.log')
        logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def _load_dataset_info(self, loader):
        """Load and cache dataset information."""
        self.anchors = loader.dataset.get_anchor_idx()
        self.poses = loader.dataset.get_pose()
        self.row_labels = loader.dataset.get_row_labels()
        self.true_loop = np.array([line == 1 for line in loader.dataset.table])
    
    def _create_param_dict(self, top_cand, roi_window, warmup_window):
        """Create parameter dictionary for tracking."""
        return {
            'top_cand': top_cand,
            'roi_window': roi_window,
            'warmup_window': warmup_window,
            'sim_func': self.sim_func,
            'save_deptrs': self.save_deptrs,
            'device': self.device,
            'dataset_name': self.run_name['seq'],
            'model_name': self.run_name['model'],
            'predictions_dir': self.predictions_dir,
            'eval_protocol': self.task,
            'monitor_range': self.monitor_range
        }
    
    def _log_configuration(self):
        """Log all configuration parameters."""
        settings = [
            ('Evaluation Settings', ''),
            ('Model', str(self.model)),
            ('Evaluation Protocol', self.task),
            ('Similarity Function', self.sim_func),
            ('Monitor Range', f'{self.monitor_range}m'),
            ('ROI Window', self.roi_window),
            ('Warmup Window', self.warmup_window),
            ('Top Candidates', self.top_cand),
            ('Dataset', self.run_name['seq']),
            ('Save Descriptors', self.save_deptrs),
            ('Device', self.device)
        ]
        
        for setting, value in settings:
            if value:
                self.logger.info(f'{setting}: {value}')
            else:
                self.logger.info(f'{setting}')
        


    def load_pretrained_model(self, checkpoint_path):
        """
        Load pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        if not os.path.isfile(checkpoint_path):
            self.logger.warning(f'Checkpoint not found: {checkpoint_path}. Generating new descriptors.')
            return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            
            # Update parameters
            self.param.update({
                'checkpoint_arch': checkpoint['arch'],
                'checkpoint_best_score': checkpoint['monitor_best']['recall'],
                'checkpoint_path': checkpoint_path
            })
            
            self.logger.info(f'Loaded model from: {checkpoint_path}')
            self.logger.info(f'Architecture: {checkpoint["arch"]}')
            self.logger.info(f'Best Score: {checkpoint["monitor_best"]["recall"]:.4f}')
        except Exception as e:
            self.logger.error(f'Failed to load checkpoint: {e}')
            raise


    def save_params(self, save_dir=None):
        """
        Save parameters to YAML file.
        
        Args:
            save_dir: Directory to save parameters (uses predictions_dir if None)
        """
        if save_dir != None:
            target_dir = self._get_save_directory(save_dir, include_protocol=True)
            os.makedirs(target_dir, exist_ok=True)
        else:
            target_dir = self.save_dir
        
        file_path = os.path.join(target_dir, 'params.yaml')
        with open(file_path, 'w') as f:
            yaml.dump(self.param, f)
        
        self.logger.info(f'Saved parameters to: {file_path}')
    
    
    
    def _get_save_directory(self, save_dir=None, include_protocol=False):
        """
        Get the target save directory.
        
        Args:
            save_dir: Custom save directory
            include_protocol: Whether to include eval_protocol in path
            
        Returns:
            Path to save directory
        """
        if save_dir is None:
            target_dir = self.predictions_dir
            if include_protocol:
                target_dir = os.path.join(target_dir, self.eval_protocol)
        else:
            target_dir = os.path.join(
                save_dir, str(self.model), self.dataset_name
            )
            if include_protocol:
                target_dir = os.path.join(target_dir, self.eval_protocol)
        
        return target_dir



    def load_descriptors(self, file=None):
        """
        Load descriptors from file.
        
        Args:
            file: Path to descriptors file (auto-finds if None)
            
        Returns:
            Loaded descriptors or None if not found
        """
        if file is None:
            files = search_files_in_dir(self.predictions_dir, 'descriptors')
            if not files:
                self.logger.warning(f'No descriptors found in {self.predictions_dir}')
                return None
            file = files[0]
        
        if not os.path.isfile(file):
            self.logger.error(f'Descriptor file not found: {file}')
            return None
        
        try:
            self.descriptors = torch.load(file, map_location=self.device)
            self.use_load_deptrs = True
            self.save_deptrs = False
            self.logger.info(f'Loaded descriptors from: {file}')
            return self.descriptors
        except Exception as e:
            self.logger.error(f'Failed to load descriptors: {e}')
            return None
    
    def save_descriptors(self, save_dir=None):
        """
        Save generated descriptors to file.
        
        Args:
            save_dir: Directory to save descriptors
        """
        # verify if the global descriptors exist
        if not hasattr(self, 'global_descriptors'):
            self.logger.error('No global descriptors found to save.')
            return None

        if self.use_load_deptrs == True:
            return None
        
        file_path = os.path.join(self.save_dir, 'descriptors.torch')
        torch.save(self.global_descriptors, file_path)
        self.logger.info(f'Saved descriptors to: {file_path}')
        return file_path
    

    def get_descriptors(self):
        '''
        Return the generated descriptors
        '''
        return self.descriptors
    
    
    def get_predictions(self):
        '''
        Return the predictions
        '''
        return self.predictions
    
    
    def load_predictions_pkl(self,file=None):
        '''
        Save the predictions in a pkl file
        params:
            file (string): file name to save the predictions, default is None
        return: None
        '''
        # Check if the results were generated
        
        # prediction is a dictionary
        # assert isinstance(self.predictions,dict), 'Predictions were not generated!'
        # Keys are ant array of integers
        if file == None:
            target_dir = os.path.join(self.predictions_dir,self.eval_protocol,self.score_value[self.monitor_range]) # Internal File name 
            file = search_files_in_dir(target_dir,'predictions.pkl') # More then one file can be found (handle this later)

            if len(file) == 0 or not os.path.isfile(file[0]): 
                self.logger.error("\n ** File does not exist: ")
                self.logger.warning("\n ** Generating predictions!")
                return None
    
            file = file[0]
        
               
        with open(file, 'rb') as handle:
            # Load the predictions
            self.predictions = pickle.load(handle)
        self.logger.warning('\n ** Loading predictions at File: ' + file)
        return self.predictions
    

    def save_predictions_pkl(self,save_dir=None):
        '''
        Save the predictions in a pkl file
        params:
            file (string): file name to save the predictions, default is None
        return: None
        '''
        # Check if the results were generated
        assert hasattr(self, 'predictions'), 'Results were not generated!'
        
        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,self.task,self.score_value[self.monitor_range]) # Internal File name 
        else:
            target_dir =  self.save_dir# os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.task,self.score_value[self.monitor_range])
            
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            self.logger.warning('\n ** Created a new directory to store predictions: ' + target_dir)
        
        
        file = os.path.join(target_dir,'predictions.pkl')
        # save predictions as a pkl file
        with open(file, 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.warning('\n ** Saving predictions at File: ' + file)
        return self.predictions
       
        
    def __save_to_csv__(self,results,file_results,res = 3):
        # SAVE global results
        colum = []
        rows  = []
        
        for value in results.items():
            keys = value[0]
            #new = keys[np.isin(keys,colum,invert=True)]
            colum.append(keys)
            rows.append(np.round(value[1],res))

        rows = np.array(rows)
        #rows = np.concatenate((top_cand,rows),axis=1)
        df = pd.DataFrame(rows.T,columns = colum)
        df.to_csv(file_results)
        
        
        
    def save_results_csv(self,save_dir=None):
        """
        Save the results in a csv file
        params:
            file (string): file name to save the results, default is None
        return: None
        """

        # Check if the results were generated
        assert hasattr(self, 'results'), 'Results were not generated!'
        if save_dir != None:
            target_dir = os.path.join(self.predictions_dir,self.eval_protocol,self.score_value[self.monitor_range]) # Internal File name 
        else:
            target_dir = self.save_dir # os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.eval_protocol,self.score_value[self.monitor_range])
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        
        self.logger.warning('\n ** Saving results from internal File: ' + target_dir)

        # SAVE Average Recall
        global_results = self.results['global']['recall']
        file_results = os.path.join(target_dir,'recall.csv')
        self.__save_to_csv__(global_results,file_results)
        self.logger.warning("Saved results at: " + file_results)
        
        
        # SAVE Average Precision
        global_results = self.results['global']['precision']
        file_results = os.path.join(target_dir,'precision.csv')
        self.__save_to_csv__(global_results,file_results)
        self.logger.warning("Saved results at: " + file_results)
        
        
        # SAVE Segment Recall
        for segment, scores in self.results['segment'].items():
            global_results = scores['recall']
            file_results = os.path.join(target_dir,f'recall_{segment}.csv')
            self.__save_to_csv__(global_results,file_results)
            self.logger.warning("Saved results at: " + file_results)
        
        
        # SAVE Segment Precision
        for segment, scores in self.results['segment'].items():
            global_results = scores['precision']
            file_results = os.path.join(target_dir,f'precision_{segment}.csv')
            self.__save_to_csv__(global_results,file_results)
            self.logger.warning("Saved results at: " + file_results)
        
        
        # SAVE Segment class Prediction performance
        if 'class' in self.results:
            class_results = self.results['class']
            file_results = os.path.join(target_dir,'class.csv')
            self.__save_to_csv__(class_results,file_results)
            self.logger.warning("Saved results at: " + file_results)



    def run(self,loop_range=10):
        
        self.loop_range_distance = loop_range
        if not isinstance(self.loop_range_distance,list):
            self.loop_range_distance = [loop_range]
        
        if self.monitor_range not in self.loop_range_distance:
            self.loop_range_distance.append(self.monitor_range)
            
        self.param['loop_range_distance'] = self.loop_range_distance
        
        # Check if the results were generated
        if not isinstance(self.top_cand,list):
            self.top_cand = [self.top_cand]


        
        self.warmup_window = 100
        self.roi_window = 50
        _distance_threshold = 2.0
        _top_k = 1
        # GROUND TRUTH
        
        # GENERATE DESCRIPTORS
        if self.use_load_deptrs == False:
            self.global_descriptors = self.generate_descriptors()

        # COMPUTE TOP 1%
        # Compute number of samples to retrieve corresponding to 1%
        n_samples = len(self.global_descriptors)
        one_percent = int(round(n_samples/100,0))
        self.top_cand.append(one_percent)
        k_top_cand = max(self.top_cand)
        
        
        # COMPUTE RETRIEVAL Performance
        # Depending on the dataset, the way datasets are split, different retrieval approaches are needed. 
        if self.task == 'relocalization':
            performance, self.predictions = eval_row_relocalization(
                                                    self.global_descriptors, # Descriptors
                                                    self.positions,   # Poses
                                                    self.row_labels, # Row labels
                                                    k_top_cand, # Max top candidates
                                                    radius=self.loop_range_distance, # Radius
                                                    roi_window=self.roi_window,
                                                    warmup_window=self.warmup_window,
                                                    sim = self.sim_func 
                                                    )
        
        elif self.task == 'place':
            performance, self.predictions = eval_row_place(self.anchors, # Anchors indices
                                                    self.global_descriptors, # Descriptors
                                                    self.positions,   # Poses
                                                    self.row_labels, # Row labels
                                                    k_top_cand, # Max top candidates
                                                    radius=self.loop_range_distance, # Radius
                                                    window=self.roi_window,
                                                    sim = self.sim_func # 
                                                    )
        else:
            raise ValueError('Wrong evaluation protocol: ' + self.eval_protocol)


        # COMPUTE Segment class Prediction performance
        content = self.global_descriptors.values()
        if 'c' in content:
            seg_preds = np.array([d['c'] for d in self.descriptors.values()])
            seg_labels = np.array([d['gt'] for d in self.descriptors.values()])
            
            class_results = compute_segment_pred(seg_preds,seg_labels)
            # update the results
            performance['class']=class_results
                    

        # Save results to be stored in csv files
        self.results = performance
        
        
        # RE-MAP TO AN OLD FORMAT
        remapped_old_format={}
        self.score_value = {}
        for range_value in self.loop_range_distance:
            remapped_old_format[range_value]={'recall':[self.results['global']['recall'][range_value][top] for  top in [0,k_top_cand-1]] }
            for segment, scores in self.results['segment'].items():
                remapped_old_format[range_value][f'recall_{segment}']= [scores['recall'][range_value][top] for  top in [0,k_top_cand-1]]           #self.logger.info(f'top {top} recall = %.3f',round(metric['recall'][25][top],3))#self.logger.info(f'top {top} recall = %.3f',round(metric['recall'][25][top],3))
        
        self.score_value[self.monitor_range] = str(round(self.results['global']['recall'][self.monitor_range][0],3)) + f'@{1}'

        return remapped_old_format


    def generate_descriptors(self):
        """
        Generate descriptors for entire dataset.
        
        Args:
            model: Neural network model
            loader: Data loader
            
        Returns:
            Dictionary of descriptors with indices as keys
        """
        self.model.eval()
        dataloader = iter(self.loader)
        num_samples = len(self.loader)
        #row_labels = self.loader.dataset.row_labels

        self.global_descriptors = {}
        
        with torch.no_grad():
            for batch_idx in tqdm(range(num_samples), desc='Generating descriptors', ncols=100):
                inputs, index = next(dataloader)
                inputs = inputs.to(self.device)
                # self.loader.dataset.struct._load_pcd_(index)
                position = self.loader.dataset.struct._get_position_(index)
                label = self.loader.dataset.struct._get_label_(index)
                # Forward pass
                outputs = self.model(inputs)
                
                # Handle different output formats
                segment_preds = None
                if isinstance(outputs, dict) and 'c' in outputs:
                    segment_preds = outputs['c']
                    descriptors = outputs['d']
                else:
                    descriptors = outputs
                
                # Validate descriptors
                assert not descriptors.isnan().any(), 'NaN values in descriptors'
                if len(descriptors.shape) < 2:
                    descriptors = descriptors.unsqueeze(0)
                
                # Convert to numpy/list
                indices_list = index.detach().cpu().numpy().tolist()
                descriptors_list = descriptors.detach().cpu().numpy().tolist()
                
                # Store predictions
                for i, (descriptor, idx) in enumerate(zip(descriptors_list, indices_list)):
                    idx = int(idx)
                    if segment_preds is None:
                        self.global_descriptors[idx] = {'d': descriptor}
                    else:
                        self.global_descriptors[idx] = {
                            'd': descriptor,
                            'c': segment_preds[i].item() if hasattr(segment_preds[i], 'item') else segment_preds[i],
                            'gt': label[idx]
                        }
        
        return self.global_descriptors



    

    

 

  
  