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
        self.logger = logger or self._create_default_logger(model, task, self.sim_func)
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

        # Build save directory path
        self.save_dir = os.path.join(self.root,
                                    run_config['save_dir'],
                                    self.run_name['experiment'],
                                    self.run_name['seq'],
                                    self.run_name['model'],
                                   )
        os.makedirs(self.save_dir, exist_ok=True)
        print("Save directory:", self.save_dir)
        
        # Dataset attributes
        #self._load_dataset_info(loader)
        
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
            'monitor_range': self.monitor_range,
            #'checkpoints': self.checkpoints
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
        """
        Save results to CSV file.
        Handles both list-based format {radius: [values]} and dict-based format {radius: {k: value}}.
        """
        colum = []
        rows  = []
        
        for key, value in results.items():
            colum.append(key)
            # Handle both dict and list/array formats
            if isinstance(value, dict):
                # New format: {k: recall_value}
                # Convert to list sorted by k values
                sorted_k = sorted(value.keys())
                row_values = [value[k] for k in sorted_k]
                rows.append(np.round(row_values, res))
            else:
                # Old format: list/array of values
                rows.append(np.round(value, res))

        rows = np.array(rows)
        df = pd.DataFrame(rows.T, columns=colum)
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


    def loop_closure_prediction(self, descriptors, labels, positions, topk, window=20) -> dict:
        """
        Evaluate descriptors for loop closure detection using similarity metrics.
        Performs retrieval on past frames only and computes predictions based on descriptor similarity.
        
        IMPORTANT RETRIEVAL RULES:
        1. Retrieval is ALWAYS done in PAST frames only (never future frames)
        2. Uses descriptor similarity (L2 or cosine) to find nearest neighbors
        3. The same past frame can be retrieved multiple times by different queries
        
        Args:
            descriptors: Dictionary of descriptors with format {idx: {'d': descriptor_vector}}
            labels: Array of segment labels for each frame
            positions: Array of 3D positions for each frame
            topk: Number of top candidates to retrieve per query
            window: Minimum frame gap to ignore immediate past frames (ROI exclusion)
            
        Returns:
            Dictionary with predictions and retrieval information:
            {
                'predictions': {query_idx: {'candidates': [...], 'similarities': [...], 'positions_dist': [...]}},
                'query_indices': [indices of query frames],
                'statistics': retrieval statistics
            }
        """
        # Extract descriptor vectors
        if isinstance(descriptors, dict) and 'd' in list(descriptors.values())[0]:
            # Extract descriptors in order of their keys
            sorted_keys = sorted(descriptors.keys())
            descriptor_array = np.array([descriptors[k]['d'] for k in sorted_keys], dtype=np.float32)
        else:
            descriptor_array = np.array(list(descriptors.values()), dtype=np.float32)
        
        n_samples = len(descriptor_array)
        all_indices = np.arange(n_samples)
        
        # Validate inputs
        assert len(labels) == n_samples, f"Labels length {len(labels)} != descriptors length {n_samples}"
        assert len(positions) == n_samples, f"Positions length {len(positions)} != descriptors length {n_samples}"
        
        predictions = {}
        query_indices = []
        
        # Ignore z-axis for position distance computation
        positions_2d = positions.copy()
        if positions_2d.ndim == 2 and positions_2d.shape[1] >= 3:
            positions_2d[:, 2] = 0
        
        self.logger.info(f'Starting loop closure prediction with topk={topk}, window={window}')
        self.logger.info(f'Total samples: {n_samples}, Similarity metric: {self.sim_func}')
        self.logger.info(f'Warmup window: {self.warmup_window}')
        
        # Process each query starting from warmup_window
        for query_idx in tqdm(range(self.warmup_window, n_samples), 
                             desc='Loop Closure Prediction', ncols=100):
            
            query_descriptor = descriptor_array[query_idx]
            query_position = positions_2d[query_idx]
            query_label = labels[query_idx]
            
            # RETRIEVAL RULE: Only consider PAST frames outside the window
            # This creates the Region of Interest (ROI) exclusion
            eligible_indices = all_indices[:query_idx - window]
            
            if len(eligible_indices) == 0:
                continue
            
            # Get eligible descriptors and positions
            eligible_descriptors = descriptor_array[eligible_indices]
            eligible_positions = positions_2d[eligible_indices]
            eligible_labels = labels[eligible_indices]
            
            # Compute descriptor similarity/distance
            if self.sim_func == 'L2':
                # Euclidean distance in descriptor space
                delta = query_descriptor - eligible_descriptors
                descriptor_distances = np.linalg.norm(delta, axis=-1)
                # Lower distance = more similar
                sort_order = np.argsort(descriptor_distances)
            elif self.sim_func == 'cosine':
                # Cosine similarity
                import torch
                from utils.loss import cosine_torch_loss
                query_tensor = torch.tensor(query_descriptor, dtype=torch.float32).unsqueeze(0)
                eligible_tensor = torch.tensor(eligible_descriptors, dtype=torch.float32)
                cosine_dist = cosine_torch_loss(query_tensor, eligible_tensor, dim=1)
                descriptor_distances = cosine_dist.cpu().numpy()
                # Flatten if needed
                if descriptor_distances.ndim > 1:
                    descriptor_distances = descriptor_distances.flatten()
                sort_order = np.argsort(descriptor_distances)
            else:
                raise ValueError(f'Invalid similarity function: {self.sim_func}')
            
            # Get top-k most similar descriptors
            topk_actual = min(topk, len(eligible_indices))
            topk_indices = sort_order[:topk_actual]
            
            # Map back to original indices
            predicted_candidates = eligible_indices[topk_indices]
            predicted_similarities = descriptor_distances[topk_indices]
            
            # Compute ground truth position distances for evaluation
            predicted_positions = eligible_positions[topk_indices]
            delta_pos = query_position - predicted_positions
            position_distances = np.linalg.norm(delta_pos, axis=-1)
            
            predicted_labels = eligible_labels[topk_indices]
            
            # Store predictions
            predictions[query_idx] = {
                'candidates': predicted_candidates.tolist(),
                'similarities': predicted_similarities.tolist(),
                'positions_dist': position_distances.tolist(),
                'labels': predicted_labels.tolist(),
                'query_label': int(query_label),
                'query_position': query_position.tolist()
            }
            
            query_indices.append(query_idx)
        
        # Compute statistics
        all_similarities = []
        all_position_dists = []
        for pred in predictions.values():
            all_similarities.extend(pred['similarities'])
            all_position_dists.extend(pred['positions_dist'])
        
        statistics = {
            'total_queries': len(query_indices),
            'topk': topk,
            'window': window,
            'similarity_metric': self.sim_func,
            'avg_descriptor_similarity': float(np.mean(all_similarities)) if all_similarities else 0.0,
            'avg_position_distance': float(np.mean(all_position_dists)) if all_position_dists else 0.0,
            'median_position_distance': float(np.median(all_position_dists)) if all_position_dists else 0.0
        }
        
        self.logger.info(f'Completed loop closure prediction for {len(query_indices)} queries')
        self.logger.info(f'Average descriptor similarity: {statistics["avg_descriptor_similarity"]:.4f}')
        self.logger.info(f'Average position distance: {statistics["avg_position_distance"]:.2f}m')
        
        return {
            'predictions': predictions,
            'query_indices': np.array(query_indices),
            'statistics': statistics,
            'parameters': {
                'topk': topk,
                'window': window,
                'warmup': self.warmup_window,
                'sim_func': self.sim_func
            }
        }

    def compute_recall_from_predictions(self, loop_closure_results: dict, 
                                         radius_thresholds: list,
                                         top_k_values: list = None) -> dict:
        """
        Compute recall performance metrics from loop_closure_prediction output.
        
        This function interfaces between the output of loop_closure_prediction
        and computes recall@k for different distance thresholds.
        
        Args:
            loop_closure_results: Output dictionary from loop_closure_prediction containing:
                - 'predictions': {query_idx: {'candidates', 'similarities', 'positions_dist', 'labels', 'query_label'}}
                - 'query_indices': array of query frame indices
                - 'statistics': retrieval statistics
                - 'parameters': retrieval parameters
            radius_thresholds: List of distance thresholds in meters for true positive detection
            top_k_values: List of top-k values to compute recall for. If None, uses [1, 5, 10, 25]
            
        Returns:
            Dictionary with performance metrics:
            {
                'global': {
                    'recall': {radius: {k: recall_value}},
                    'precision': {radius: {k: precision_value}},
                    'num_queries': int
                },
                'segment': {
                    segment_id: {
                        'recall': {radius: {k: recall_value}},
                        'precision': {radius: {k: precision_value}},
                        'num_queries': int
                    }
                }
            }
        """
        predictions = loop_closure_results['predictions']
        
        if top_k_values is None:
            top_k_values = [1, 5, 10, 25]

        top_k_range = range(1, max(top_k_values) + 1)

        # Ensure radius_thresholds is a list
        if not isinstance(radius_thresholds, list):
            radius_thresholds = [radius_thresholds]
            
        # Initialize result containers
        # Global metrics
        global_tp = {r: {k: 0 for k in top_k_range} for r in radius_thresholds}
        global_total = {r: {k: 0 for k in top_k_range} for r in radius_thresholds}
        
        # Segment-wise metrics
        segments = set()
        for pred in predictions.values():
            segments.add(pred['query_label'])
        
        segment_tp = {seg: {r: {k: 0 for k in top_k_range} for r in radius_thresholds} for seg in segments}
        segment_total = {seg: {r: {k: 0 for k in top_k_range} for r in radius_thresholds} for seg in segments}
        
        # Process each query prediction
        for query_idx, pred in predictions.items():
            query_label = pred['query_label']
            position_distances = np.array(pred['positions_dist'])
            candidate_labels = np.array(pred['labels'])
            
            # For each radius threshold
            for radius in radius_thresholds:
                # For each top-k value
                for k in top_k_range:
                    # Get top-k predictions
                    topk_dists = position_distances[:k] if len(position_distances) >= k else position_distances
                    topk_labels = candidate_labels[:k] if len(candidate_labels) >= k else candidate_labels
                    
                    # Check if any of top-k predictions is a true positive
                    # True positive: position distance <= radius AND same segment label
                    tp_mask = (topk_dists <= radius) & (topk_labels == query_label)
                    is_tp = np.any(tp_mask)
                    
                    # Update global counters
                    global_tp[radius][k] += int(is_tp)
                    global_total[radius][k] += 1
                    
                    # Update segment counters
                    segment_tp[query_label][radius][k] += int(is_tp)
                    segment_total[query_label][radius][k] += 1
        
        # Compute recall values
        global_recall = {}
        global_precision = {}
        for radius in radius_thresholds:
            global_recall[radius] = {}
            global_precision[radius] = {}
            for k in top_k_range:
                if global_total[radius][k] > 0:
                    global_recall[radius][k] = global_tp[radius][k] / global_total[radius][k]
                    global_precision[radius][k] = global_tp[radius][k] / (global_total[radius][k] * k)
                else:
                    global_recall[radius][k] = 0.0
                    global_precision[radius][k] = 0.0
        
        # Compute segment-wise recall
        segment_results = {}
        for seg in segments:
            segment_results[seg] = {
                'recall': {},
                'precision': {},
                'num_queries': sum(segment_total[seg][radius_thresholds[0]][k] for k in [top_k_range[0]])
            }
            for radius in radius_thresholds:
                segment_results[seg]['recall'][radius] = {}
                segment_results[seg]['precision'][radius] = {}
                for k in top_k_range:
                    if segment_total[seg][radius][k] > 0:
                        segment_results[seg]['recall'][radius][k] = segment_tp[seg][radius][k] / segment_total[seg][radius][k]
                        segment_results[seg]['precision'][radius][k] = segment_tp[seg][radius][k] / (segment_total[seg][radius][k] * k)
                    else:
                        segment_results[seg]['recall'][radius][k] = 0.0
                        segment_results[seg]['precision'][radius][k] = 0.0
        
        # Log segment-wise recall@1 for the first radius
        primary_radius = radius_thresholds[0] if radius_thresholds else 10
        for seg in sorted(segments):
            recall_at_1 = segment_results[seg]['recall'].get(primary_radius, {}).get(1, 0.0)
            print(f'Segment: {seg}: {recall_at_1}')
        
        return {
            'global': {
                'recall': global_recall,
                'precision': global_precision,
                'num_queries': len(predictions)
            },
            'segment': segment_results
        }

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


        # PERFORM LOOP CLOSURE PREDICTION using descriptors
        loop_closure_results = self.loop_closure_prediction(
            self.global_descriptors, 
            self.row_labels,
            self.positions,
            k_top_cand,
            window=self.roi_window
        )
        
        # Store loop closure predictions for later analysis
        self.loop_closure_results = loop_closure_results
        
        # COMPUTE RETRIEVAL Performance using new interface
        # Build top_k list from self.top_cand (convert indices to k values)
        top_k_for_recall = sorted(list(set(self.top_cand)))  # [1, 5, 25, one_percent, ...]
        
        performance = self.compute_recall_from_predictions(
            loop_closure_results,
            radius_thresholds=self.loop_range_distance,
            top_k_values=top_k_for_recall
        )
        
        # Store predictions from loop_closure_results for compatibility
        self.predictions = loop_closure_results['predictions']


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
        
        
        # RE-MAP TO AN OLD FORMAT for backwards compatibility
        remapped_old_format={}
        self.score_value = {}
        for range_value in self.loop_range_distance:
            # Get recall values - use keys from the top_k_for_recall list
            recall_values = []
            for k in [top_k_for_recall[0], top_k_for_recall[-1]]:  # first and last k values
                recall_values.append(self.results['global']['recall'][range_value].get(k, 0.0))
            remapped_old_format[range_value] = {'recall': recall_values}
            
            for segment, scores in self.results['segment'].items():
                segment_recall_values = []
                for k in [top_k_for_recall[0], top_k_for_recall[-1]]:
                    segment_recall_values.append(scores['recall'][range_value].get(k, 0.0))
                remapped_old_format[range_value][f'recall_{segment}'] = segment_recall_values
        
        # Get recall@1 for the monitor range
        recall_at_1 = self.results['global']['recall'].get(self.monitor_range, {}).get(top_k_for_recall[0], 0.0)
        self.score_value[self.monitor_range] = str(round(recall_at_1, 3)) + f'@{top_k_for_recall[0]}'

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



    

    

 

  
  