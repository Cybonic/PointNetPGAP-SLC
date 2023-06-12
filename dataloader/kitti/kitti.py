

from dataloader.kitti.kitti_eval import KITTIEval
from dataloader.kitti.kitti_triplet import KittiTriplet
from torch.utils.data import DataLoader,SubsetRandomSampler
import numpy as np

class KITTI():
    def __init__(self,**kwargs):

        self.kwargs = kwargs
        self.root =  kwargs.pop('root')
        self.val_cfg   = kwargs.pop('val_loader')
        self.train_cfg = kwargs['train_loader']
        self.modality  = kwargs['modality']
        self.max_points = kwargs['max_points']

    def get_train_loader(self,debug=True):
        sequence  = self.train_cfg['sequence']
        #max_points = self.max_points
        

        train_loader = KittiTriplet( root = self.root,
                                    sequences = sequence,
                                    modality = self.modality,
                                    ground_truth = self.train_cfg['ground_truth']
                                                )
        
        if debug == False:
            trainloader   = DataLoader(train_loader,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = self.train_cfg['shuffle'],
                                    num_workers= 0,
                                    pin_memory=False,
                                    drop_last=True,
                                    )
        else:
            indices = np.random.randint(0,len(train_loader),20)
            np.random.shuffle(indices)
            sampler = SubsetRandomSampler(indices)

            trainloader   = DataLoader(train_loader,
                                    batch_size = 1, #train_cfg['batch_size'],
                                    shuffle    = False,
                                    num_workers= 0,
                                    pin_memory=False,
                                    drop_last=True,
                                    sampler=sampler
                                    )
        
        return trainloader
    
    def get_test_loader(self):
        raise NotImplementedError
    
    
    def get_val_loader(self):
        sequence  = self.val_cfg['sequence']
        

        val_loader = KITTIEval( root = self.root,
                               sequence = sequence[0],
                               modality = self.modality,
                             #**self.val_cfg,
                                ground_truth = self.val_cfg['ground_truth']
                                )

        valloader  = DataLoader(val_loader,
                                batch_size = self.val_cfg['batch_size'],
                                num_workers= 0,
                                pin_memory=False,
                                )
        
        return valloader
    
    def get_label_distro(self):
        raise NotImplemented
		#return  1-np.array(self.label_disto)