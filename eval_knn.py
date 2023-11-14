#!/usr/bin/env python3

import argparse
import yaml
import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import torch 
from tqdm import tqdm
import numpy as np
from networks import contrastive
from utils.eval import eval_row_place
import logging
import pandas as pd

from utils.utils import get_available_devices
from pipeline_factory import model_handler,dataloader_handler


class PlaceRecognition():
    def __init__(self,model,
                    loader,
                    top_cand,
                    window,
                    eval_metric,
                    logger,
                    loop_range_distance = 10,
                    save_deptrs=True,
                    device='cpu',
                    **arg):

        self.loop_range_distance = loop_range_distance
        self.eval_metric = eval_metric
        self.logger = logger
        if device in ['gpu','cuda']:
            device, availble_gpus = get_available_devices(1,logger)
        
        self.device = device
        self.model = model.to(self.device)
        self.loader = loader
     

        self.top_cand = top_cand
        self.window   = window
  
        self.model_name = str(self.model).lower()
        self.save_deptrs= save_deptrs # Save descriptors after being generated 
        self.use_load_deptrs= False # Load descriptors when they are already generated 

        # Eval data
      
        self.dataset_name = str(loader.dataset)
        self.database = loader.dataset.get_idx_universe()
        self.anchors  = loader.dataset.get_anchor_idx()
        table = loader.dataset.table
        self.poses = loader.dataset.get_pose()
        self.raw_labels = loader.dataset.get_row_labels()

        self.true_loop = np.array([line==1 for line in table])

        # SAVE PREDICTIONS
        self.predictions_dir = os.path.join('saved_model_data',arg['logdir'],f'{str(self.model)}',f'{self.dataset_name}')
        #self.prediction_file = os.path.join(predictions_dir,f'{str(self.model)}')
        if not os.path.isdir(self.predictions_dir):
            os.makedirs(self.predictions_dir)
            logger.warning('\n ** Created a new directory: ' + self.predictions_dir)
        
        self.param = {}
        self.param['top_cand'] = top_cand
        self.param['window'] = window
        self.param['eval_metric'] = eval_metric
        self.param['loop_range_distance'] = loop_range_distance
        self.param['save_deptrs'] = save_deptrs
        self.param['device'] = device
        self.param['dataset_name'] = self.dataset_name
        self.param['model_name'] = self.model_name
        self.param['predictions_dir'] = self.predictions_dir




    def load_pretrained_model(self,checkpoint_path):
        '''
        Load the pretrained model
        params:
            checkpoint_path (string): path to the pretrained model
        return: None
        '''
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.param['checkpoint_arch'] = checkpoint['arch']
        self.param['checkpoint_best_score'] = checkpoint['monitor_best']['recall']
        self.param['checkpoint_path'] = checkpoint_path

        
        self.logger.warning('\n ** Loaded pretrained model from: ' + checkpoint_path)
        self.logger.warning('\n ** Architecture: ' + checkpoint['arch'])
        self.logger.warning('\n ** Best Score: %0.2f'%checkpoint['monitor_best']['recall'])

    def save_params(self):
        """
        Save the parameters of the model
        params:

        return: None
        """

        target_dir = os.path.join(self.predictions_dir,self.score_value[10])
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        
        file_name = os.path.join(target_dir,'params.yaml')
        with open(file_name, 'w') as file:
            documents = yaml.dump(self.param, file)
        self.logger.warning('\n ** Saving parameters at File: ' + file_name)



    def load_descriptors(self,file):
        """
        Load descriptors from a file
        params:
            file (string): file name to load the descriptors
        return: None
        """
        
        target_dir = os.path.join(self.predictions_dir,file)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        
        file = os.path.join(target_dir,'descriptors.torch')
            
        if not os.path.isfile(file): 
            self.logger.error("\n ** File does not exist: "+ file)
            self.logger.warning("\n ** Generating descriptors!")
            return
        
        else:
            self.logger.warning('\n ** Loading descriptors from internal File: ' + file)

        # LOADING DESCRIPTORS
        self.descriptors = torch.load(file)

        self.use_load_deptrs = True # Disable descriptor generation
        self.save_deptrs = False # Descriptors were already saved, so no need to save again
    


    def save_descriptors(self,file=None):
        '''
        Save the generated descriptors
        params:
            file (string): file name to save the descriptors, default is None
        return: None
        '''

        if file == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[10])
        else:
            target_dir = os.path.join(self.predictions_dir,file)
        
        # Create directory
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
        file = os.path.join(target_dir,'descriptors.torch')

        # LOADING DESCRIPTORS
        if self.save_deptrs == True:
            torch.save(self.descriptors,file)
            self.logger.warning('\n ** Saving descriptors at File: ' + file)
    

    def get_descriptors(self):
        '''
        Return the generated descriptors
        '''
        return self.descriptors
        

    def save_predictions_cv(self,file=None):
        '''
        Save the predictions in a csv file
        params:
            file (string): file name to save the predictions, default is None
        return: None
        '''
        # Check if the results were generated
        assert  hasattr(self, 'predictions')
        
        loop_cand   = self.predictions['loop_cand']
        loop_scores = self.predictions['loop_scores']
        gt_loops    = self.predictions['gt_loops']

        assert hasattr(self, 'score_value'), 'Results were not generated!'
        
        if file == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[10],'place') # Internal File name 
        else:
            target_dir = os.path.join(self.predictions_dir,file,'place') # Internal File name 
        
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

        

    def save_results_cv(self,file=None):
        """
        Save the results in a csv file
        params:
            file (string): file name to save the results, default is None
        return: None
        """

        # Check if the results were generated
        assert hasattr(self, 'results'), 'Results were not generated!'
        if file == None:
            target_dir = os.path.join(self.predictions_dir,self.score_value[10],'place') # Internal File name 
        else:
            target_dir = os.path.join(self.predictions_dir,file,'place') # Internal File name 
        
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


    def run(self,loop_range= None):
        
        if isinstance(loop_range,list):
            self.loop_range_distance = loop_range
        if not isinstance(self.loop_range_distance,list):
            self.loop_range_distance = [self.loop_range_distance]

        if not isinstance(self.top_cand,list):
            self.top_cand = [self.top_cand]
        
        # GENERATE DESCRIPTORS
        if self.use_load_deptrs == False:
            self.descriptors = self.generate_descriptors(self.model,self.loader)
        
        # COMPUTE TOP 1%
        # Compute number of samples to retrieve correspondin to 1% 
        one_percent = int(round(len(self.database)/100,0))
        self.top_cand.append(one_percent)
        k_top_cand = max(self.top_cand)

        # COMPUTE RETRIEVAL
        # Depending on the dataset, the way datasets are split, different retrieval approaches are needed. 
        # the kitti dataset 
        # radius = 2
        raw_labels = self.raw_labels
        metric, self.predictions = eval_row_place(self.anchors, # Anchors indices
                                                  self.descriptors, # Descriptors
                                                  self.poses,   # Poses
                                                  raw_labels, # Raw labels
                                                  k_top_cand, # Max top candidates
                                                  radius=self.loop_range_distance, # Radius
                                                  window=self.window)

        # RE-MAP TO AN OLD FORMAT
        remapped_old_format={}
        self.score_value = {}
        for range_value in self.loop_range_distance:
            remapped_old_format[range_value]={'recall':[metric['recall'][range_value][top] for  top in [0,k_top_cand-1]] }            #self.logger.info(f'top {top} recall = %.3f',round(metric['recall'][25][top],3))
            self.score_value[range_value] = str(round(metric['recall'][range_value][0],3)) + f'@{1}'
        self.results = metric

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

        #self._reset_metrics()
        prediction_bag = {}
        for batch_idx in tbar:
            input,inx = next(dataloader)
            input = input.to(self.device)
            input = input.type(torch.cuda.FloatTensor)
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




if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")

    parser.add_argument(
        '--network', '-m',
        type=str,
        required=False,
        default='PointNetVLAD',
        help='Directory to get the trained model.'
    )
    
    parser.add_argument(
        '--experiment', '-e',
        type=str,
        required=False,
        default='remove',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--cfg', '-f',
        type=str,
        required=False,
        default='sensor-cfg',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--resume', '-p',
        type=str,
        required=False,
        default='None',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--memory',
        type=str,
        required=False,
        default='Disk',
        choices=['Disk','RAM'],
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--debug', '-b',
        type=bool,
        required=False,
        default=False,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--plot',
        type=int,
        required=False,
        default=1,
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--modality',
        type=str,
        required=False,
        default='pcl',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='uk',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        required=False,
        default='[orchards/june23/extracted]',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--device',
        type=str,
        required=False,
        default='cuda',
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=False,
        default=10,
        help='Directory to get the trained model.'
    )
    parser.add_argument(
        '--max_points',
        type=int,
        required=False,
        default = 500,
        help='sampling points.'
    )
    parser.add_argument(
        '--ground_truth_file',
        type=str,
        required=False,
        default = "ground_truth_ar1m_nr10m_pr2m.pkl",
        help=' ground truth file.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    ###################################################################### 
    
     # The development has been made on different PC, each has some custom settings
    # e.g the root path to the dataset;
    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]

    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions', FLAGS.dataset.lower() + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

    device_name = os.uname()[1]
    pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
    root_dir = pc_config[device_name]


    print("----------")
    print("INTERFACE:")
    print("Root: ", root_dir)
    print("Memory: ", FLAGS.memory)
    print("Model:  ", FLAGS.network)
    print("Debug:  ", FLAGS.debug)
    print("Resume: ", FLAGS.resume)
    print(f'Device: {FLAGS.device}')
    print(f'batch size: {FLAGS.batch_size}')
    print("----------\n")
    ###################################################################### 

    # DATALOADER
    SESSION['max_points']= FLAGS.max_points
    SESSION['retrieval']['top_cand'] = list(range(1,25,1))
    SESSION['val_loader']['ground_truth_file'] = FLAGS.ground_truth_file

    # Build the model and the loader
    model  = model_handler(FLAGS.network, num_points=SESSION['max_points'],output_dim=256)
    loader = dataloader_handler(root_dir,FLAGS.network,FLAGS.dataset,SESSION)

    logger = logging.getLogger("Knn Eval")

    pl = PlaceRecognition(model, # Model
                         loader.get_val_loader(), # Get the Test loader
                        25, # Max retrieval Candidates
                        600, # Warmup
                        'L2', # Similarity Metric
                        logger, # Logger
                        device  = FLAGS.device # Device
                        ) # ,windows,eval_metric,device
    
    # Load the descriptors if they exist
    pl.load_descriptors('temp')
    # Run place recognition evaluation
    pl.run()
    
    # Save the results
    pl.save_descriptors('temp')
    pl.save_predictions_cv('temp')
    pl.save_results_cv('temp')

    

    

 

  
  