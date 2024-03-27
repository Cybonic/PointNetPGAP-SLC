


import os
import numpy as np
from tqdm import tqdm
import numpy as np
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor
import torch


class CollationFunctionFactory:
    """
    Taken from LoGG3DNet package
    
    """
    def __init__(self, collation_type='default', voxel_size=0.05, num_points=80000):
        self.voxel_size = voxel_size
        self.num_points = num_points
        if collation_type == 'default':
            self.collation_fn = self.collate_default
        elif collation_type == 'tuple':
            self.collation_fn = self.collate_tuple
        elif collation_type == 'sparse':
            self.collation_fn = self.collate_sparse
        elif collation_type == 'sparse_tuple':
            self.collation_fn = self.collate_sparse_tuple
        elif collation_type == 'torch_tuple':
            self.collation_fn = self.collate_torch_tuple
        elif collation_type == 'reg_sparse_tuple':
            self.collation_fn = self.collate_reg_sparse_tuple
        elif collation_type == 'sparcify_list':
            self.collation_fn = self.sparcify_and_collate_list
        else:
            raise ValueError(f'collation_type {collation_type} not found')

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        #if len(list_data) > 1:
        return self.collate_torch(list_data)

    def collate_torch(self, list_data):
        outputs_pts = []
        output_idx  = []
        for i,(batch_data) in enumerate(list_data):
            for tuple_data in batch_data:
                if torch.is_tensor(tuple_data):
                    outputs_pts.append(tuple_data)
                elif isinstance(tuple_data, (np.int64,int)):
                    output_idx.append(tuple_data)
            #outputs.extend(contrastive_tuple)
        #outputs = [torch.from_numpy(ct).float() for ct in contrastive_tuple]
        outputs = torch.stack(outputs_pts)
        return outputs,torch.tensor(output_idx)
    

    def collate_tuple(self, list_data):
        outputs = []
        for batch_data in list_data:
            contrastive_tuple = []
            for tuple_data in batch_data:
                if isinstance(tuple_data, np.ndarray):
                    contrastive_tuple.append(tuple_data)
                elif isinstance(tuple_data, list):
                    contrastive_tuple.extend(tuple_data)
            # outputs.append(sparse_collate(contrastive_tuple))
        outputs = [torch.from_numpy(ct).float() for ct in contrastive_tuple]
        outputs = torch.stack(outputs)
        return outputs

    def collate_sparse(self, list_data):
        """
        Batching of Sparse-based methods
        Input: [(sparse,numpy int64), ...]
        Output: sparse, list of ints

        """
        outputs = []
        sparse_list = []
        id_list = []
        for batch_data in list_data:
            for data in batch_data:
                if isinstance(data, SparseTensor):
                    sparse_list.append(data)
                elif isinstance(data, (list, np.ndarray,np.int64,int)):
                    id_list.append(data)
        outputs = sparse_collate(sparse_list)
        outputs.C = outputs.C.int()
        return outputs,torch.tensor(id_list)
        
    def collate_sparse_tuple(self, list_data):
        assert len(list_data)==1

        #outputs = []
        sparse_list = []
        sparse_name = []
        sparse_data = []
        indices_list = []
        labels = []
        contrastive_tuple = []
        for i,tuple_data in enumerate(list_data):
            
            # indice 0 - data
            # indice 1 - indices of the data in the dataset

            if i>0: # currently only support one tuple
                break

            for name in tuple_data[0].keys():
                if name in "anchor":
                    sparse_list.extend([tuple_data[0][name]])
                    sparse_name.append(name)
                if name in  ['positive','negative']:
                    if isinstance(tuple_data[0][name][0], SparseTensor):
                        n_elem = len(tuple_data[0][name])
                        sparse_name.extend([name]*n_elem)
                        sparse_list.extend(tuple_data[0][name])

            labels.extend(sparse_name)

            for name in tuple_data[1].keys():
                if name in "anchor":
                    indices_list.extend([tuple_data[1][name]])
                else:
                    indices_list.extend(tuple_data[1][name])

        sparse_data = sparse_collate(sparse_list)
        return sparse_data,indices_list,labels
    

    def collate_torch_tuple(self, list_data):
        assert len(list_data)==1

        #outputs = []
        contrastive_tuple = []
        for tuple_data in list_data[0]:
            for name in tuple_data.keys():
                if name in  ['positive','negative']:
                    if isinstance(tuple_data[name][0], torch.Tensor):
                        tuple_data[name] = torch.stack(tuple_data[name])
            contrastive_tuple.append(tuple_data)
        return contrastive_tuple

    def collate_reg_sparse_tuple(self, list_data):
        outputs = []
        for tuple_data in list_data:
            contrastive_tuple = []
            meta_info = None
            for name in tuple_data.keys():
                if isinstance(tuple_data[name], SparseTensor):
                    contrastive_tuple.append(tuple_data[name])
                elif isinstance(tuple_data[name], (list, np.ndarray)):
                    contrastive_tuple.extend(tuple_data[name])
                elif isinstance(tuple_data[name], dict):
                    meta_info = tuple_data[name]
            outputs.append([sparse_collate(contrastive_tuple), meta_info])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def sparcify_and_collate_list(self, list_data):
        outputs = []
        if isinstance(list_data, SparseTensor):
            return list_data
        else:
            # return outputs
            for xyzr in list_data:
                xyzr = xyzr[0]
                if not len(xyzr) > 0:
                    continue
                pc_ = np.round(xyzr[:, :3] / self.voxel_size).astype(np.int32)
                pc_ -= pc_.min(0, keepdims=1)
                feat_ = xyzr

                _, inds, inverse_map = sparse_quantize(pc_,
                                                       return_index=True,
                                                       return_inverse=True)
                if len(inds) > self.num_points:
                    inds = np.random.choice(
                        inds, self.num_points, replace=False)

                pc = pc_[inds]
                feat = feat_[inds]
                outputs.append(SparseTensor(feat, pc))
            return sparse_collate(outputs)


def load_sync_indices(file):
    overlap = []
    for f in open(file):
        f = f.split(':')[-1]
        indices = [int(i) for i in f.split(' ')]
        overlap.append(indices)
    return(np.array(overlap[0]))


def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([f.split('.')[0] for f in os.listdir(target_dir)])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])
    return(files[idx],fullfiles[idx])



def gen_ground_truth(   poses,
                        pos_range= 0.05, # Loop Threshold [m]
                        neg_range= 10,
                        num_neg =  10,
                        num_pos =  10,
                        warmupitrs= 10, # Number of frames to ignore at the beguinning
                        roi       = 5 # Window):
                    ):

    indices = np.array(range(poses.shape[0]-1))
    
    ROI = indices[warmupitrs:]
    anchor   = []
    positive = []

    for i in ROI:
        _map_   = poses[:i-roi,:]
        pose    = poses[i,:].reshape((1,-1))
        dist_meter  = np.sqrt(np.sum((pose -_map_)**2,axis=1))

        pos_idx = np.where(dist_meter < pos_range)[0]

        if len(pos_idx)>=num_pos:
            pos_dist = dist_meter[pos_idx]
            min_sort = np.argsort(pos_dist)
            
            if num_pos == -1:
                pos_select = pos_idx[min_sort]
            else:
                pos_select = pos_idx[min_sort[:num_pos]]

            positive.append(pos_select)
            anchor.append(i)
    # Negatives
    negatives= []
    neg_idx = np.arange(1,num_neg)   
    for a, pos in zip(anchor,positive):
        pa = poses[a,:].reshape((1,-1))
        dist_meter = np.linalg.norm((pa-poses),axis=-1)
        neg_idx =np.array([i for i,data in enumerate(dist_meter) if data > neg_range and not i in pos])
        select_neg = np.random.randint(0,len(neg_idx),num_neg)
        neg_idx = neg_idx[select_neg]
        neg_= np.linalg.norm(poses[a,:].reshape((1,-1))-poses[neg_idx,:], axis=-1)
        assert all(neg_>neg_range)
        negatives.append(neg_idx)

    return(anchor,positive,negatives)



def make_data_loader(root_dir,dataset,session,modality,max_points=50000):

    dataset = dataset.lower()
    assert dataset in ['kitti','orchard-uk','uk','pointnetvlad'],'Dataset Name does not exist!'
    
    
    #if dataset == 'kitti':
        # Kitti 
    from dataloader.kitti.kitti import KITTI as DATALOADER
    
    loader = DATALOADER( root = root_dir,
                    dataset = dataset,
                    modality = modality,
                    memory   = session['memory'],
                    train_loader  = session['train_loader'],
                    val_loader    = session['val_loader'],
                    max_points    = session['max_points']
                    )
    
    #elif dataset in ['orchards-uk','uk'] :

    #    from .ORCHARDS import ORCHARDS

    #    loader = ORCHARDS(  root    = root_dir,
    #                        modality = modality,
    #                        train_loader  = session['train_loader'],
    #                        test_loader   = session['val_loader'],
    #                        memory = session['memory'],
    #                        max_points    = session['max_points']
    #                        )
    
    
    #elif dataset == 'pointnetvlad':
        
    #    from .POINTNETVLAD import POINTNETVLAD
        
    #    loader = POINTNETVLAD(root       = session[root_dir],
    #                        train_loader = session['train_loader'],
    #                        val_loader   = session['val_loader'],
    #                        memory       = memory
    #                        )
    

    #elif dataset == 'fuberlin':
        
        #session['train_loader']['root'] =  session[root_dir]
    #    session['val_loader']['root'] =  session[root_dir]
    #    loader = FUBERLIN(
    #                        train_loader  = session['train_loader'],
    #                        val_loader    = session['val_loader'],
    #                        mode          = memory,
    #                        max_points = 50000
    #                        )
    
    return(loader)
