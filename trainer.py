
from cmath import nan
from base.base_trainer import BaseTrainer
from tqdm import tqdm  
import numpy as np
from evaluator import PlaceRecognition
import logging

# ===================================================================================================================
#       
#
#
# ===================================================================================================================

class Trainer(BaseTrainer):
    def __init__(self,  model,
                        train_loader,
                        test_loader,
                        config,
                        resume = None,
                        device = 'cpu',
                        run_name = 'default',
                        train_epoch_zero = True,
                        monitor_range = 1, # The range to monitor the performance (meters)
                        eval_protocol='place'
                        
                        ):

        super(Trainer, self).__init__(model, resume, config, run_name=run_name, device=device, train_epoch_zero=train_epoch_zero)


        self.trainer_cfg    = config
        self.train_loader   = train_loader
        self.test_loader     = test_loader
        self.val_loader    = None
        self.device         = device
        self.model          = model.to(self.device)
        self.hyper_log      = config
        self.loss_dist      = config['loss']['args']['metric']
        
        logger = logging.getLogger("Training")
        
        self.eval_metric = config['trainer']['eval_metric']
        self.top_cand_retrieval = config['retrieval']['top_cand']
        
        assert isinstance(self.top_cand_retrieval,list)

        self.train_metrics = None #StreamSegMetrics(len(labels))
        self.val_metrics = None #StreamSegMetrics(len(labels))
        
        
        window = 600 # Avoid the nearby frames
        self.monitor_range = monitor_range
        self.eval_approach = PlaceRecognition(self.model ,
                                                self.test_loader,
                                                self.top_cand_retrieval,
                                                window,
                                                self.loss_dist,
                                                logger,
                                                device= self.device,
                                                eval_protocol = eval_protocol,
                                                logdir =  run_name['experiment'],
                                                monitor_range = monitor_range
                                                )

# ===================================================================================================================
# 
# ===================================================================================================================
    def mean_grad(self,batch_size):

        for name, param in self.model.named_parameters():
            if param.requires_grad and  param.grad is not None:
                
                param.grad /= batch_size
    
    def _train_epoch(self, epoch, batch_size = 10 ):
        
        self.logger.info('\n')
        self.model.train()
     
        dataloader = iter(self.train_loader)
        tbar = tqdm(range(len(self.train_loader)), ncols=80)

        self._reset_metrics()
        epoch_loss_list = {}
        epoch_loss = 0
        
        self.optimizer.zero_grad()
        for batch_idx in tbar:
            
            input = next(dataloader)
                    
            batch_data ,info= self.model(input)
            
            for key,value in info.items():
                if key in epoch_loss_list:
                    epoch_loss_list[key].append(value.detach().cpu().item())
                else:
                    epoch_loss_list[key] = [value.detach().cpu().item()]
   
            # Accumulate error
            epoch_loss += batch_data.detach().cpu().item()

            # Update progress bar description
            tbar.set_description('T ({}) | Loss {:.10f}'.format(epoch,epoch_loss/(batch_idx+1)))
            tbar.update()

            # Backward and optimize
            if batch_idx % batch_size == 0 and batch_idx > 0:
                # Update model every batch_size iteration
                self.mean_grad(batch_size)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        # Update model every batch_size iteration
        epoch_perfm = {}
        for key,value in epoch_loss_list.items():
            epoch_perfm[key] = np.mean(value)
        
        epoch_perfm['loss'] = epoch_loss/batch_idx
        self._write_scalars_tb('train',epoch_perfm,epoch)

        return epoch_perfm

            
# ===================================================================================================================
#    
# ===================================================================================================================

    def _valid_epoch(self,epoch,loop_range = None):

        if loop_range is None or isinstance(loop_range,int):
            loop_range = [loop_range]
        
        # Evaluate the model
        overall_scores = self.eval_approach.run(loop_range=loop_range)

        max_cand = self.top_cand_retrieval[-1]
        
        # Post on tensorboard
        for range,scores in overall_scores.items():
            for score,top in zip(scores['recall'],['1','1%']):
                self._write_scalars_tb(f'Recall val@{top}',{f'Range {range}':score},epoch)
        
        # For model comparison use the first range top 1 recall
        output = {'recall':overall_scores[self.monitor_range]['recall'][0]}
        return output,[]

# ===================================================================================================================
#    
# ===================================================================================================================
    def _write_scalars_tb(self,wrt_mode,logs,epoch):
        for k, v in logs.items():
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
            #if 'mIoU' not in k: self.writer.add_scalar(f'train/{k}', v, self.wrt_step)
        if 'train' in wrt_mode:
            for i, opt_group in enumerate(self.optimizer.param_groups):
                self.writer.add_scalar(f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
    

    def _write_hyper_tb(self,logs):
        # https://towardsdatascience.com/a-complete-guide-to-using-tensorboard-with-pytorch-53cb2301e8c3        
        hparam_dict = { "batch_size": self.hyper_log['train_loader']['batch_size'],
                        "experim_name": str(self.hyper_log['experim_name']),
                        "dataset": str(self.hyper_log['val_loader']['dataset']),
                        "sequence": str(self.hyper_log['val_loader']['sequence']),
                        "model": self.hyper_log['modelwrapper']['type'],
                        "minibatch_size": self.hyper_log['modelwrapper']['minibatch_size'],
                        #"output_dim": self.hyper_log['model']['output_dim'],
                        "optim": str(self.hyper_log['modelwrapper']['type']),
                        "lr": self.hyper_log['optimizer']['args']['lr'],
                        "wd": self.hyper_log['optimizer']['args']['weight_decay'],
                        "lr_scheduler": self.hyper_log['optimizer']['lr_scheduler'],
        }

        metric_dict = logs
        #self.writer.add_hparams(hparam_dict,metric_dict)
                        
    
