#!/usr/bin/env python3

import argparse
import yaml
import os
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import torch 
from tqdm import tqdm
import numpy as np
from networks import contrastive
from utils.retrieval import retrieval_knn
from utils.metric import retrieve_eval
from utils.eval import eval_place
import logging
import pandas as pd

from dataloader import utils
from utils.utils import get_available_devices
from utils import loss as loss_lib
from pipeline_factory import pipeline


def retrieval_costum_dist(queries,database,descriptors,top_cand,window,metric,**argv):
    '''
    Retrieval function 
    
    '''

    database_dptrs   = np.array([descriptors[i] for i in database])
  

    scores,winner = [],[]
    for query in tqdm(queries,"Retrieval"):
        query_dptrs = database_dptrs[query]

        dist = np.array([metric(query_dptrs,database_dptrs[i]) for i in range(database_dptrs.shape[0])])
        ind = np.argsort(dist)
        # Remove query index
        idx_window = np.arange(query-window,len(database))
        remove = [ind[0] == i for i in idx_window]
        remove_bool = np.invert(np.sum(remove,axis=0,dtype=np.bool8))

        # remove_idx = -(np.sum(remove,axis=0)-1)
  
        scores.append(dist[0][remove_bool])
        winner.append(ind[0][remove_bool])

        #retrieved_loops,scores = retrieval_knn(query_dptrs, database_dptrs, top_cand =top, metric = eval_metric)
    return(np.array(winner),np.array(scores))



def retrieval_sequence(queries,database,descriptors,top_cand,window,metric,**argv):
    '''
    Retrieval function 
    
    '''

    from sklearn.neighbors import KDTree
    
    
    database_dptrs   = np.array([descriptors[i] for i in database])
    tree = KDTree(database_dptrs.squeeze(), leaf_size=2)

    scores,winner = [],[]
    for query in tqdm(queries,"Retrieval"):
        query_dptrs = database_dptrs[query]
        query_dptrs = query_dptrs.reshape(1,-1)
        dist, ind = tree.query(query_dptrs, k=top_cand)
        # Remove query index
        idx_window = np.arange(query-window,len(database))
        remove = [ind[0] == i for i in idx_window]
        remove_bool = np.invert(np.sum(remove,axis=0,dtype=np.bool8))

        # remove_idx = -(np.sum(remove,axis=0)-1)
  
        scores.append(dist[0][remove_bool])
        winner.append(ind[0][remove_bool])

        #retrieved_loops,scores = retrieval_knn(query_dptrs, database_dptrs, top_cand =top, metric = eval_metric)
    return(winner,scores)



def retrieval(queries,database,descriptors,top_cand,metric,**argv):
    '''
    Retrieval function 
    
    '''
    #metric = 'euclidean' if metric == 'L2'
    from sklearn.neighbors import KDTree
    
    database_dptrs   = np.array([descriptors[i] for i in database])
    tree = KDTree(database_dptrs.squeeze(), leaf_size=2)

    scores,winner = [],[]
    for query in tqdm(queries,"Retrieval"):
        query_dptrs = database_dptrs[query]
        query_dptrs = query_dptrs.reshape(1,-1)
        dist, ind = tree.query(query_dptrs, k=top_cand)
        # Remove query index
        values = np.invert(query == ind[0])

        scores.append(dist[0][values])
        winner.append(ind[0][values])

        #retrieved_loops,scores = retrieval_knn(query_dptrs, database_dptrs, top_cand =top, metric = eval_metric)
    return(np.array(winner),np.array(scores))

class PlaceRecognition():
    def __init__(self,model,loader,top_cand,window,eval_metric,logger,save_deptrs=True,device='cpu'):

        self.eval_metric = eval_metric
        self.logger = logger
        if device in ['gpu','cuda']:
            device, availble_gpus = get_available_devices(1,logger)
        
        self.device = device
        self.model = model.to(self.device)
        self.loader = loader
        self.loader.dataset.todevice(self.device)
        self.top_cand = top_cand
        self.window = window
        #self.pred_loops=[]
        #self.pred_scores=[]
        self.model_name = str(self.model).lower()
        self.save_deptrs= save_deptrs # Save descriptors after being generated 
        self.use_load_deptrs= False # Load descriptors when they are already generated 

        # Eval data
        try:
            self.dataset_name = str(loader.dataset)
            self.database = loader.dataset.get_idx_universe()
            self.anchors = loader.dataset.get_anchor_idx()
            table = loader.dataset.get_gt_map()
            #
            self.poses = loader.dataset.get_pose()
        except: 
            dataset_name = str(loader.dataset.dataset)
            self.database = loader.dataset.dataset.get_idx_universe()
            self.anchors = loader.dataset.dataset.get_anchor_idx()
            table = loader.dataset.dataset.get_gt_map()
            #poses = loader.dataset.dataset.get_pose()

        #self.true_loop = np.array([np.where(line==1)[0] for line in table])
        #self.true_loop = [np.where(line==1)[0] for line in table]
        self.true_loop = np.array([line==1 for line in table])
        #self.true_loop = np.ones_like(self.true_loop_bool)*-1
        #for line in self.true_loop_bool:
        #    self.true_loop[line==true] 

        # SAVE DESCRIPTORS
        self.descriptors_dir = os.path.join('predictions',f'{self.dataset_name}','descriptors')
        self.descriptor_file = os.path.join(self.descriptors_dir,f'{str(self.model)}.torch') 
        if self.save_deptrs==True and not os.path.isdir(self.descriptors_dir):
            os.makedirs(self.descriptors_dir)
            logger.warning('\n ** Created a new directory: ' + self.descriptors_dir)
        
        # SAVE RESULTS
        results_dir = os.path.join('predictions',f'{self.dataset_name}','place','results')
        self.results_file = os.path.join(results_dir,f'{str(self.model)}')
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
            logger.warning('\n ** Created a new directory: ' + results_dir)
        
        # SAVE PREDICTIONS
        predictions_dir = os.path.join('predictions',f'{self.dataset_name}','place','loops')
        self.prediction_file = os.path.join(predictions_dir,f'{str(self.model)}')
        if not os.path.isdir(predictions_dir):
            os.makedirs(predictions_dir)
            logger.warning('\n ** Created a new directory: ' + predictions_dir)
        

    def load_descriptors(self,file=None):
        '''
        Load previously generated descriptors.

        Input: path to the descriptor file
         When file == None, use internal build file name,
           which is based on the model and dataset name  
        '''
        if file == None:
            file = self.descriptor_file # Internal File name 
            
        if not os.path.isfile(file): 
            self.logger.error("\n ** File does not exist: "+ file)
            self.logger.warning("\n ** Generating descriptors!")
            return
        
        else:
            self.logger.warning('\n ** Loading descriptors from internal File: ' + self.descriptor_file)

        # LOADING DESCRIPTORS
        self.descriptors = torch.load(file)
        self.use_load_deptrs = True # Disable descriptor generation
        self.save_deptrs = False # Descriptors were already saved, so no need to save again


    def get_descriptors(self):
        '''
        Return the generated descriptors
        '''
        return self.descriptors
        

    def save_predictions_cv(self,file=None):
        # SAVE PREDICTIONS
        # Check if the predictions were generated
        assert  hasattr(self, 'pred_loops') and \
                hasattr(self, 'pred_scores') and   \
                hasattr(self, 'target_loops'), 'Predictions were not generated!'
        
        assert hasattr(self, 'score_value'), 'Results were not generated!'
        
        if file == None:
            file = self.prediction_file # Internal File name 
            self.logger.warning('\n ** Saving results from internal File: ' + self.prediction_file)

        df_pred   = pd.DataFrame(self.pred_loops)
        df_score  = pd.DataFrame(self.pred_scores)
        df_target = pd.DataFrame(self.target_loops)
    
        file_results_ped = self.prediction_file + f'_loops_{self.score_value}.csv'
        file_results_score =self.prediction_file + f'_scores_{self.score_value}.csv'
        file_target =self.prediction_file + f'_target_{self.score_value}.csv'

        df_pred.to_csv(file_results_ped)
        df_score.to_csv(file_results_score)
        df_target.to_csv(file_target)

        

    def save_results_cv(self,file=None):
        
        # Check if the results were generated
        assert hasattr(self, 'results'), 'Results were not generated!'
        if file == None:
            file = self.results_file # Internal File name 
            logger.warning('\n ** Saving results from internal File: ' + self.results_file)

        top_cand = np.array(list(self.results.keys())).reshape(-1,1)
        colum = ['top']
        rows  = []
        for value in self.results.items():
            keys = np.array(list(value[1].keys()))
            new = keys[np.isin(keys,colum,invert=True)]
            colum.extend(new)
            rows.append(list(value[1].values()))

        rows = np.array(rows)
        rows = np.concatenate((top_cand,rows),axis=1)
        df = pd.DataFrame(rows,columns = colum)
        file_results = file + '_' + self.score_value +'.csv'
        df.to_csv(file_results)
        self.logger.warning("Saved results at: " + file_results)


    def run(self):
        
        if not isinstance(self.top_cand,list):
            self.top_cand = [self.top_cand]
        
        # GENERATE DESCRIPTORS
        if self.use_load_deptrs == False:
            self.descriptors = self.generate_descriptors(self.model,self.loader)
        
        # SAVE DESCRIPTORS
        if self.save_deptrs:
            torch.save(self.descriptors,self.descriptor_file)
            self.logger.info("Descriptors saved at: " + self.descriptor_file)

        # COMPUTE TOP 1%
        # Compute number of samples to retrieve correspondin to 1% 
        one_percent = int(round(len(self.database)/100,0))
        self.top_cand.append(one_percent)
        max_top = max(self.top_cand)

        # COMPUTE RETRIEVAL
        # Depending on the dataset, the way datasets are split, different retrieval approaches are needed. 
        # the kitti dataset 
        metric,predictions = eval_place(self.anchors,self.descriptors,self.poses,max_top,window=self.window)

        # SAVE PREDICTIONS

        # RE-MAP TO AN OLD FORMAT
        remapped_old_format={}
        for top in tqdm(range(1,max_top),'Performance'):
            remapped_old_format[top]={'recall':metric['recall'][25][top]}
            #self.logger.info(f'top {top} recall = %.3f',round(metric['recall'][25][top],3))

        self.score_value = str(round(metric['recall'][25][0],3)) + f'@{1}'
        self.results = remapped_old_format

        return remapped_old_format


    def generate_descriptors(self,model,loader):
            model.eval()
            dataloader = iter(loader)
            num_sample = len(loader)
            tbar = tqdm(range(num_sample), ncols=100)

            #self._reset_metrics()
            prediction_bag = {}
            idx_bag = []
            for batch_idx in tbar:
                input,inx = next(dataloader)
                input = input.to(self.device)
                # Generate the Descriptor
                prediction,feat = model(input)
                assert prediction.isnan().any() == False
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
        default='scancontext',
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
        default='kitti',
        help='Directory to get the trained model.'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        required=False,
        default='[00]',
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

    FLAGS, unparsed = parser.parse_known_args()

    ###################################################################### 
    # LOAD DEFAULT SESSION PARAM
    session_cfg_file = os.path.join('sessions', FLAGS.dataset.lower() + '.yaml')
    print("Opening session config file: %s" % session_cfg_file)
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))


    #from networks import modeling
    #model_ = modeling.__dict__[FLAGS.network]()
    

    print("----------")
    print("INTERFACE:")
    print("Root: ", SESSION['root'])
    print("Memory: ", FLAGS.memory)
    print("Model:  ", FLAGS.network)
    print("Debug:  ", FLAGS.debug)
    print("Resume: ", FLAGS.resume)
    print(f'Device: {FLAGS.device}')
    print(f'batch size: {FLAGS.batch_size}')
    print("----------\n")
    ###################################################################### 

    # DATALOADER
    model_,dataloader = pipeline(FLAGS.network,FLAGS.dataset,SESSION)
    model_wrapper = contrastive.ModelWrapper(model_,**SESSION['modelwrapper'])
    #dataloader = utils.load_dataset(FLAGS.dataset,SESSION,FLAGS.memory)                            
    
    SESSION['train_loader']['data']['max_points'] = FLAGS.max_points
    SESSION['val_loader']['data']['max_points'] = FLAGS.max_points
    SESSION['retrieval']['top_cand'] = list(range(1,25,1))

    logger = logging.getLogger("Knn Eval")

    pl = PlaceRecognition(model_,
                        dataloader.get_val_loader(), # Get the Test loader
                        25, # Max retrieval Candidates
                        600, # Warmup
                        'L2', # Similarity Metric
                        logger, # Logger
                        save_deptrs = True,
                        device  = FLAGS.device # Device
                        ) # ,windows,eval_metric,device
    
    # Load the descriptors if they exist
    pl.load_descriptors()
    # Run place recognition evaluation
    pl.run()
    pl.save_predictions_cv()
    pl.save_results_cv()

    

    

 

  
  