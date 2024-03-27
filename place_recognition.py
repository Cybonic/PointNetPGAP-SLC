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


def search_files_in_dir(directory,search_file):
    files_found = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith(search_file):
                files_found.append(os.path.join(root, file))
    return files_found
            
class PlaceRecognition():
    def __init__(self,model,
                    loader,
                    top_cand,
                    eval_metric,
                    logger,
                    roi_window    = 600,
                    warmup_window = 100,
                    save_deptrs   = True,
                    device        = 'cpu',
                    eval_protocol = 'place',
                    monitor_range = 1, # m
                    **arg):

        self.monitor_range = monitor_range
        self.eval_protocol = eval_protocol
        self.eval_metric   = eval_metric
        self.logger = logger
        
        if device in ['gpu','cuda']:
            device, availble_gpus = get_available_devices(1,logger)
        
        self.device = device
        self.model  = model#.to(self.device)
        self.loader = loader
        self.top_cand = top_cand
        self.roi_window    = roi_window
        self.warmup_window = warmup_window
        
        self.model_name      = str(self.model).lower()
        self.save_deptrs     = save_deptrs # Save descriptors after being generated 
        self.use_load_deptrs = False # Load descriptors when they are already generated 


        self.dataset_name = str(loader.dataset)
        self.anchors  = loader.dataset.get_anchor_idx()
        table      = loader.dataset.table
        self.poses = loader.dataset.get_pose()
        self.row_labels = loader.dataset.get_row_labels()

        self.true_loop = np.array([line==1 for line in table])

        # SAVE PREDICTIONS
        self.predictions_dir = os.path.join('saved_model_data',arg['logdir'],f'{str(self.model)}',f'{self.dataset_name}')
        #self.prediction_file = os.path.join(predictions_dir,f'{str(self.model)}')
        if not os.path.isdir(self.predictions_dir):
            os.makedirs(self.predictions_dir)
            logger.warning('\n ** Created a new directory: ' + self.predictions_dir)
        
        self.param = {}
        self.param['top_cand']     = top_cand
        self.param['roi_window']   = self.roi_window
        self.param['warmup_window'] = self.warmup_window
        self.param['eval_metric']  = eval_metric
        self.param['save_deptrs']  = save_deptrs
        self.param['device']       = device
        self.param['dataset_name'] = self.dataset_name
        self.param['model_name']   = self.model_name
        self.param['predictions_dir'] = self.predictions_dir
        self.param['eval_protocol'] = self.eval_protocol
        self.param['monitor_range'] = self.monitor_range
        


    def load_pretrained_model(self,checkpoint_path):
        '''
        Load the pretrained model
        params:
            checkpoint_path (string): path to the pretrained model
        return: None
        '''
        if not os.path.isfile(checkpoint_path):
            
            print("\n ** File does not exist: "+ checkpoint_path)
            self.logger.warning("\n ** Generating descriptors!")
            return None
        
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.param['checkpoint_arch'] = checkpoint['arch']
        self.param['checkpoint_best_score'] = checkpoint['monitor_best']['recall']
        self.param['checkpoint_path'] = checkpoint_path
        
        self.logger.warning('\n ** Loaded pretrained model from: ' + checkpoint_path)
        self.logger.warning('\n ** Architecture: ' + checkpoint['arch'])
        self.logger.warning('\n ** Best Score: %0.2f'%checkpoint['monitor_best']['recall'])


    def save_params(self,save_dir=None):
        """
        Save the parameters of the model
        params:

        return: None
        """
        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[self.monitor_range])
        else:
            target_dir = os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.score_value[self.monitor_range])

        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            print('\n ** Created a new directory: ' + target_dir)
            self.logger.warning('\n ** Created a new directory: ' + target_dir)
        
        file_name = os.path.join(target_dir,'params.yaml')
        with open(file_name, 'w') as file:
            documents = yaml.dump(self.param, file)
        self.logger.warning('\n ** Saving parameters at File: ' + file_name)



    def load_descriptors(self,file=None):
        """
        Load descriptors from a file
        params:
            file (string): file name to load the descriptors
        return: None
        """
        #target_dir = os.path.join(self.predictions_dir)
        if file == None:
            target_dir = self.predictions_dir
            #target_dir = os.path.join(self.predictions_dir,file)
            file = search_files_in_dir(target_dir,'descriptors')[0]
        
        if not os.path.isfile(file): 
            self.logger.error("\n ** File does not exist: "+ file)
            self.logger.warning("\n ** Generating descriptors!")
            return
        
        else:
            self.logger.warning('\n ** Loading descriptors from internal File: ' + file)

        # LOADING DESCRIPTORS
        self.descriptors = torch.load(file)
        self.use_load_deptrs = True # Disable descriptor generation
        self.save_deptrs     = False # Descriptors were already saved, so no need to save again
    


    def save_descriptors(self,save_dir=None,):
        '''
        Save the generated descriptors
        params:
            file (string): file name to save the descriptors, default is None
        return: None
        '''

        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[self.monitor_range])
        else:
            target_dir = os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.score_value[self.monitor_range])
        
        # Create directory
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            print('\n ** Created a new directory: ' + target_dir)
            self.logger.warning('\n ** Created a new directory: ' + target_dir)
        file = os.path.join(target_dir,'descriptors.torch')

        # LOADING DESCRIPTORS
        torch.save(self.descriptors,file)
        self.logger.warning('\n ** Saving descriptors at File: ' + file)
    

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
    
    
    def load_predictions_pkl(self,save_dir=None):
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
        predictions_dir = os.path.join(self.predictions_dir,f"{self.monitor_range}m")
        file = search_files_in_dir(predictions_dir,'predictions.pkl') # More then one file can be found (handle this later)
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
        assert  hasattr(self, 'predictions')
        
        # prediction is a dictionary
        assert isinstance(self.predictions,dict), 'Predictions were not generated!'
        # Keys are ant array of integers
        assert isinstance(list(self.predictions.keys())[0].item(),int), 'Predictions were not generated!'
      
        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,f"{self.monitor_range}m",self.score_value[self.monitor_range],self.eval_protocol) # Internal File name 
        else:
            target_dir = os.path.join(save_dir,f'{str(self.model)}',f"{self.monitor_range}m",self.monitor_range,self.score_value[self.monitor_range],self.eval_protocol)
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            self.logger.warning('\n ** Created DIR at: ' +target_dir)
        
        file = os.path.join(target_dir,'predictions.pkl')
        # save predictions as a pkl file
        with open(file, 'wb') as handle:
            pickle.dump(self.predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.logger.warning('\n ** Saving predictions at File: ' + file)
        return self.predictions

        
            
            
    def save_predictions_cv(self,save_dir=None):
        '''
        Save the predictions in a csv file
        params:
            file (string): file name to save the predictions, default is None
        return: None
        '''
        # Check if the results were generated
        assert  hasattr(self, 'predictions')
        
        assert 'loop_cand' in self.predictions.keys(), 'Predictions were not generated!'
        loop_cand   = self.predictions['loop_cand']
        loop_scores = self.predictions['loop_scores']
        gt_loops    = self.predictions['gt_loops']

        assert hasattr(self, 'score_value'), 'Results were not generated!'
        
        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[self.monitor_range],self.eval_protocol) # Internal File name 
        else:
            target_dir = os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.score_value[self.monitor_range],self.eval_protocol)
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        self.logger.warning('\n ** Saving predictions at: ' +target_dir)

        df_pred   = pd.DataFrame(loop_cand)
        df_score  = pd.DataFrame(loop_scores)
        df_target = pd.DataFrame(gt_loops)
    
        file_results_ped = os.path.join(target_dir,f'loops.csv')
        file_results_score =os.path.join(target_dir,f'scores.csv')
        file_target =os.path.join(target_dir,f'target.csv')

        df_pred.to_csv(file_results_ped)
        df_score.to_csv(file_results_score)
        df_target.to_csv(file_target)

        

    def save_results_cv(self,save_dir=None):
        """
        Save the results in a csv file
        params:
            file (string): file name to save the results, default is None
        return: None
        """

        # Check if the results were generated
        assert hasattr(self, 'results'), 'Results were not generated!'
        if save_dir == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[self.monitor_range],'place') # Internal File name 
        else:
            target_dir = os.path.join(save_dir,f'{str(self.model)}',f'{self.dataset_name}',self.score_value[self.monitor_range],self.eval_protocol)
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        
        self.logger.warning('\n ** Saving results from internal File: ' + target_dir)

        top_cand = np.array(list(self.results.keys())).reshape(-1,1)
        colum = []
        rows  = []
        
        for value in self.results['recall'].items():
            keys = value[0]
            #new = keys[np.isin(keys,colum,invert=True)]
            colum.append(keys)
            rows.append(np.round(value[1],3))

        rows = np.array(rows)
        #rows = np.concatenate((top_cand,rows),axis=1)
        df = pd.DataFrame(rows.T,columns = colum)
        file_results = os.path.join(target_dir,'results_recall.csv')
        df.to_csv(file_results)
        self.logger.warning("Saved results at: " + file_results)
        
        colum = []
        rows  = []
        for value in self.results['precision'].items():
            keys = value[0]
            #new = keys[np.isin(keys,colum,invert=True)]
            colum.append(keys)
            rows.append(np.round(value[1],3))

        rows = np.array(rows)
        #rows = np.concatenate((top_cand,rows),axis=1)
        df = pd.DataFrame(rows.T,columns = colum)
        file_results = os.path.join(target_dir,'results_precision.csv')
        df.to_csv(file_results)
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
        
        # Check if the results were generated       
        sim_func =  'sc_similarity' if str(self.model).startswith("scancontext") else 'L2'
        
        # GENERATE DESCRIPTORS
        if self.use_load_deptrs == False:
            self.descriptors = self.generate_descriptors(self.model,self.loader)
        
        # COMPUTE TOP 1%
        # Compute number of samples to retrieve corresponding to 1% 
        n_samples = len(self.descriptors)
        one_percent = int(round(n_samples/100,0))
        self.top_cand.append(one_percent)
        k_top_cand = max(self.top_cand)
        
        
        
        
        # COMPUTE RETRIEVAL
        # Depending on the dataset, the way datasets are split, different retrieval approaches are needed. 
        if self.eval_protocol == 'relocalization':
            metric, self.predictions = eval_row_relocalization(
                                                    self.descriptors, # Descriptors
                                                    self.poses,   # Poses
                                                    self.row_labels, # Row labels
                                                    k_top_cand, # Max top candidates
                                                    radius=self.loop_range_distance, # Radius
                                                    roi_window=self.roi_window,
                                                    warmup_window=self.warmup_window,
                                                    sim = sim_func 
                                                    )
        
        elif self.eval_protocol == 'place':
            metric, self.predictions = eval_row_place(self.anchors, # Anchors indices
                                                    self.descriptors, # Descriptors
                                                    self.poses,   # Poses
                                                    self.row_labels, # Row labels
                                                    k_top_cand, # Max top candidates
                                                    radius=self.loop_range_distance, # Radius
                                                    window=self.roi_window,
                                                    sim = sim_func # 
                                                    )
        else:
            raise ValueError('Wrong evaluation protocol: ' + self.eval_protocol)

        # RE-MAP TO AN OLD FORMAT
        remapped_old_format={}
        self.score_value = {}
        for range_value in self.loop_range_distance:
            remapped_old_format[range_value]={'recall':[metric['recall'][range_value][top] for  top in [0,k_top_cand-1]] }            #self.logger.info(f'top {top} recall = %.3f',round(metric['recall'][25][top],3))
            
        self.results = metric
        
        self.score_value[self.monitor_range] = str(round(metric['recall'][self.monitor_range][0],3)) + f'@{1}'

        return remapped_old_format


    def generate_descriptors(self,model,loader):
        '''
        Generate descriptors for the whole dataset
        params:
            model: network model
            loader: data loader
        return:
            descriptors: descriptors for the whole dataset
        '''
        
        model.eval()
        dataloader = iter(loader)
        num_sample = len(loader)
        tbar = tqdm(range(num_sample), ncols=100)

        prediction_bag = {}
        for batch_idx in tbar:
            input,inx = next(dataloader)
            input = input.to(self.device)
            # Generate the Descriptor
            prediction = model(input)
            #prediction,feat = model(input)
            assert prediction.isnan().any() == False
            if len(prediction.shape)<2:
                prediction = prediction.unsqueeze(0)
            # Keep descriptors
            for d,i in zip(prediction.detach().cpu().numpy().tolist(),inx.detach().cpu().numpy().tolist()):
                prediction_bag[int(i)] = d
        return(prediction_bag)



    

    

 

  
  