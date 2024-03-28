import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np


class SparseModelWrapper(nn.Module):
    def __init__(self,  model,
                        loss        = None,
                        minibatch_size = 3, 
                        device = 'cuda',
                        **args,
                        ):
                        
        super(SparseModelWrapper,self).__init__()
        assert minibatch_size>= 3, 'Minibatch size too small'
        
        self.loss = loss
        self.minibatch_size = minibatch_size
        self.device = device
        self.batch_counter = 0 
        self.model = model
        self.device = 'cpu'
    
        try:
            self.device =  next(self.parameters()).device
        except:
            print('ModelWrapper device: ',self.device)
        

    def forward(self,pcl,**arg):
        self.device =  next(self.parameters()).device
        self.model.to(self.device)
        # Mini Batch training due to memory constrains
        
        if self.training == False:
            pred = self.model(pcl.to(self.device))
            pred = pred['out']
            return(pred)

        # Training
        # Adaptation to new paradigm
        batch_loss = 0
        sparse_data = pcl[0].to(self.device)  # pcl[0] is the sparse tensor
        sparse_index = np.array(pcl[1]) # pcl[1] is the sparse index
        labels = pcl[2]       # pcl[2] is the labels

        anchor_idx = np.array([idx for idx,label in enumerate(labels) if label == 'anchor'])
        positive_idx = np.array([idx for idx,label in enumerate(labels) if label == 'positive'])
        negative_idx = np.array([idx for idx,label in enumerate(labels) if label == 'negative'])
        
        pred = self.model(sparse_data.to(self.device))
        
        feat = pred['feat']
        pred = pred['out']
            
        descriptor = {'a':pred[anchor_idx],'p':pred[positive_idx],'n':pred[negative_idx]}
        poses = {'a':sparse_index[anchor_idx],'p':sparse_index[positive_idx],'n':sparse_index[negative_idx]}
        
        # Triplet Loss calculation
        loss_value, info = self.loss(descriptor = descriptor, poses = poses)
        loss_value.backward() # Back propagate gradients and free graph
        batch_loss += loss_value

        return(batch_loss,info)

    def __str__(self):
        out = [str(self.model),
               str(self.loss)
               ]
        out = '-'.join(out)
        return out
    
    
    
    
class SparseModelWrapperLoss(nn.Module):
    def __init__(self,  model,
                        loss        = None,
                        minibatch_size = 3, 
                        device = 'cuda',
                        aux_loss_on = 'pairloss',
                        representation = 'descriptors',
                        class_loss_margin = 0.1,
                        **args,
                        ):
                        
        super(SparseModelWrapperLoss,self).__init__()
        assert minibatch_size>= 3, 'Minibatch size too small'
        
        self.loss = loss
        self.minibatch_size = minibatch_size
        self.device = device
        self.batch_counter = 0 
        self.model             = model
        self.pooling           = args['pooling']
        
        self.class_loss_margin = class_loss_margin
        self.device = 'cpu'
        
        self.loss_on = aux_loss_on
        
        if self.loss_on == 'pairloss':
            self.sec_loss = pcl_binary_loss(**args)
        elif self.loss_on == 'segmentloss':
            self.sec_loss = segment_loss()
        else:
            print('No loss specified')   
 
        self.representation = representation
            
        try:
            self.device =  next(self.parameters()).device
        except:
            print('ModelWrapper device: ',self.device)


    def forward(self,pcl,**arg):
        self.device =  next(self.parameters()).device
        self.model.to(self.device)
        # Mini Batch training due to memory constrains
        if self.training == False:
            
            pred = self.model(pcl.to(self.device))
            pred = pred['out']
            # Check for NaN
            assert not torch.any(torch.isnan(pred), dim=0).any()
            return(pred)

        # Training
        # Adaptation to new paradigm
        batch_loss = 0
        sparse_data = pcl[0].to(self.device)  # pcl[0] is the sparse tensor
        sparse_index = np.array(pcl[1]) # pcl[1] is the sparse index
        labels = pcl[2]       # pcl[2] is the labels

        anchor_idx = np.array([idx for idx,label in enumerate(labels) if label == 'anchor'])
        positive_idx = np.array([idx for idx,label in enumerate(labels) if label == 'positive'])
        negative_idx = np.array([idx for idx,label in enumerate(labels) if label == 'negative'])
        
        pred = self.model(sparse_data.to(self.device))
        
        feat = pred['feat']
        pred = pred['out']
        # Check for NaN
        assert not torch.any(torch.isnan(pred), dim=0).any()
            
        descriptor = {'a':pred[anchor_idx],'p':pred[positive_idx],'n':pred[negative_idx]}
        poses = {'a':sparse_index[anchor_idx],'p':sparse_index[positive_idx],'n':sparse_index[negative_idx]}
        
        # Triplet Loss calculation
        loss_value, info = self.loss(descriptor = descriptor, poses = poses)
        
        if self.loss_on == 'pairloss':
            if self.representation == 'features':
                if self.pooling == 'max':
                    feat = torch.max(feat, dim=1, keepdim=False)[0]
                elif self.pooling == 'mean':
                    feat = torch.mean(feat, dim=1, keepdim=False)
                else:
                    raise ValueError('Pooling not defined')
                    
                da,dp,dn = feat[anchor_idx],feat[positive_idx],feat[negative_idx]
            else:
                da,dp,dn = pred[anchor_idx],pred[positive_idx],pred[negative_idx]
            
            
            class_loss_value = self.sec_loss(da,dp,dn)
            loss_value =  self.class_loss_margin * loss_value + (1-self.class_loss_margin)*class_loss_value
            info['class_loss'] = class_loss_value.detach()
        
        
        elif self.loss_on == 'segmentloss':    
            #if 'labels' in arg:
            self.row_labels = torch.tensor(arg['labels'],dtype=torch.float32).to(self.device)
            #pred[anchor_idx],pred[positive_idx],pred[negative_idx]
            target = torch.cat((self.row_labels[sparse_index[anchor_idx]],self.row_labels[sparse_index[positive_idx]],self.row_labels[sparse_index[negative_idx]]))
                
            class_loss_value = self.sec_loss( pred, target)
            loss_value =  self.class_loss_margin * loss_value + (1-self.class_loss_margin)*class_loss_value
            info['class_loss'] = class_loss_value.detach()
        

        loss_value.backward() # Backpropagate gradients and free graph
        batch_loss += loss_value

        return(batch_loss,info)

    def __str__(self):
        
        out = [str(self.model),
               str(self.loss)]
        if self.loss_on in ['pairloss','segmentloss']:
            out.append(f'{self.sec_loss}M{self.class_loss_margin}')
        if self.loss_on == 'pairloss':
            out.append(f'{self.representation}P{self.pooling}')
        #out= np.squeeze(out).tolist()
        out = '-'.join(out)
        return out
    


class ModelWrapper(nn.Module):
    def __init__(self,  model,
                        loss        = None,
                        minibatch_size = 3, 
                        **args,
                        ):
                        
        super(ModelWrapper,self).__init__()
        assert minibatch_size>= 3, 'Minibatch size too small'
        
        self.loss = loss
        self.minibatch_size = minibatch_size
        #self.device = device
        self.batch_counter = 0 
        self.model = model
        self.device = 'cpu'
        
        
        try:
            self.device =  next(self.parameters()).device
        except:
            print('ModelWrapper device: ',self.device)
    
        print('ModelWrapper device: ',self.device)
        

    def forward(self,pcl,**argv):
        
        self.device =  next(self.parameters()).device
        self.model.to(self.device)
        # Mini Batch training due to memory constrains
        if self.training == False:
            pred = self.model(pcl.to(self.device)) # pred = self.model(pcl.cuda())
            return(pred)

        # Training
        # Adaptation to new paradigm
        anchor,positive,negative = pcl[0]['anchor'],pcl[0]['positive'],pcl[0]['negative']
        pose_anchor,pose_positive,pose_negative = pcl[1]['anchor'],pcl[1]['positive'],pcl[1]['negative']
        num_anchor,num_pos,num_neg = 1,len(positive),len(negative)
        

        if len(anchor.shape)<4:
            anchor = anchor.unsqueeze(0)
        if positive.shape[0]>1:
            positive = torch.cat(positive)
        
        pose = {'a':pose_anchor,'p':pose_positive,'n':pose_negative}
        
        batch_loss = 0
        mini_batch_total_iteration = math.ceil(num_neg/self.minibatch_size)
        for i in range(0,mini_batch_total_iteration): # This works because neg < pos
            
            j = i*self.minibatch_size
            k = j + self.minibatch_size
            if k > num_neg:
                k = j + (num_neg - j)

            neg = negative[j:k]

            pclt = torch.cat((anchor,positive,neg))
            
            if pclt.shape[0]==1: # drop last
                continue

            pred = self.model(pclt.to(self.device)) # pclt.cuda()
            
            a_idx = num_anchor
            p_idx = num_pos+num_anchor
            n_idx = num_pos+num_anchor + num_neg
            
            dq,dp,dn = pred[0:a_idx],pred[a_idx:p_idx],pred[p_idx:n_idx]
            descriptor = {'a':dq,'p':dp,'n':dn}

            loss_value,info = self.loss(descriptor = descriptor, poses = pose)
            # devide by the number of batch iteration; as direct implication in the grads
            loss_value /= mini_batch_total_iteration 
            
            loss_value.backward() # Backpropagate gradients and free graph
            batch_loss += loss_value

        return(batch_loss,info)
    

    def __str__(self):
        return str(self.model) + '-' + str(self.loss)
    
    

class ModelWrapperLoss(nn.Module):
    def __init__(self,  model,
                        loss        = None,
                        minibatch_size = 3, 
                        margin = 0.5,
                        **args,
                        ):
                        
        super(ModelWrapperLoss,self).__init__()
        assert minibatch_size>= 3, 'Minibatch size too small'
        
        self.loss = loss
        self.minibatch_size = minibatch_size
        #self.device = device
        self.batch_counter = 0 
        self.model = model
        self.device = 'cpu'
        self.margin = margin
        self.representation = args['representation']
        self.class_loss_on = args['class_loss_on']
        self.pooling = args['pooling']
        
        self.sec_loss = pcl_binary_loss() 
        print('ModelWrapper device: ',self.device)


    def forward(self,pcl,**argv):
        
        self.device =  next(self.parameters()).device
        if self.training == False:
            pred = self.model(pcl.to(self.device)) # pred = self.model(pcl.cuda())
            pred = pred['out']
            return(pred)

        # Training
        row_labels = argv['labels']
        # Adaptation to new paradigm
        anchor,positive,negative = pcl[0]['anchor'],pcl[0]['positive'],pcl[0]['negative']
        pose_anchor,pose_positive,pose_negative = pcl[1]['anchor'],pcl[1]['positive'],pcl[1]['negative']
        num_anchor,num_pos,num_neg = 1,len(positive),len(negative)
        
        labels = []
        labels.append(row_labels[pose_anchor])
        labels.extend(row_labels[pose_positive])
        labels.extend(row_labels[pose_negative])
        
        if len(anchor.shape)<4:
            anchor = anchor.unsqueeze(0)
        if positive.shape[0]>1:
            positive = torch.cat(positive)
        
        pose = {'a':pose_anchor,'p':pose_positive,'n':pose_negative}
        
        batch_loss = 0
        mini_batch_total_iteration = math.ceil(num_neg/self.minibatch_size)
        
        for i in range(0,mini_batch_total_iteration): # This works because neg < pos
            
            j = i*self.minibatch_size
            k = j + self.minibatch_size
            if k > num_neg:
                k = j + (num_neg - j)

            neg = negative[j:k]

            pclt = torch.cat((anchor,positive,neg))
            
            if pclt.shape[0]==1: # drop last
                continue
            
            pred = self.model(pclt.to(self.device)) # pclt.cuda()
            
            feat = pred['feat']
            pred = pred['out'] 
            
            global_loss = 0
            
            a_idx = num_anchor
            p_idx = num_pos+num_anchor
            n_idx = num_pos+num_anchor + num_neg
            
            dq,dp,dn = pred[0:a_idx],pred[a_idx:p_idx],pred[p_idx:n_idx]
            descriptor = {'a':dq,'p':dp,'n':dn}

            if self.class_loss_on:
                if self.representation == 'features':
                    if self.pooling == 'max':
                        feat = torch.max(feat, dim=1, keepdim=False)[0]
                    elif self.pooling == 'mean':
                        feat = torch.mean(feat, dim=1, keepdim=False)[0]
                    da,dp,dn = feat[0:a_idx],feat[a_idx:p_idx],feat[p_idx:n_idx]
                else:
                    da,dp,dn = pred[0:a_idx],pred[a_idx:p_idx],pred[p_idx:n_idx]
                
            class_loss_value = self.sec_loss(da,dp,dn)
            loss_value,info = self.loss(descriptor = descriptor, poses = pose)
            loss_value += self.margin*class_loss_value
            info['class_loss'] = class_loss_value.detach()

            
            # devide by the number of batch iteration; as direct implication in the grads
            loss_value /= mini_batch_total_iteration 
            
            loss_value.backward() # Backpropagate gradients and free graph
            batch_loss += loss_value

        return(batch_loss,info)
    
    def __str__(self):
        
        name = [str(self.model),
                str(self.loss),
               f'{self.sec_loss}M{self.margin}' if self.class_loss_on else 'noloss',
               self.representation if self.class_loss_on else '',
                self.pooling if self.class_loss_on else ''
               ]
        str_name = '-'.join(name)
        return str_name




def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        #layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers    



class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out
    


class segment_loss(nn.Module):
    def __init__(self,num_c=7, feat_dim=256, w=0.1):
        super().__init__()
        self.w = w
        
        list_layers = mlp_layers(feat_dim, [256, 64], b_shared=False, bn_momentum=0.01, dropout=0.0)
        list_layers.append(torch.nn.Linear(64, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)
    
    def forward(self, descriptor, target):
        
        target = target.long()
        correct_targets = target > -1
        target = target[correct_targets]
        descriptor = descriptor[correct_targets]
        
        out = self.classifier(descriptor)
        loss_c = F.nll_loss(F.log_softmax(out, dim=1), target,reduction='mean')
        #loss_c = self.classloss(
        #    torch.nn.functional.log_softmax(out, dim=1), target)
        return loss_c
    def __str__(self):
        return "segment_loss"
    


class pcl_binary_loss(torch.nn.Module):
    def __init__(self,in_dim=512,kernels= [256,64],**argv):
        super().__init__()
        #self.margin = margin
        self.fc = MLPNet(in_dim,kernels,b_shared=False, bn_momentum=0.0, dropout=0.0)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        last_size = kernels[-1]
        self.last_layer = torch.nn.Linear(last_size,1)
        
    def forward(self, anchor,positive,negative):
        # concatenate anchors and positives
        anchor = anchor.unsqueeze(1)
        positive = positive.unsqueeze(1)
        negative = negative.unsqueeze(1)
        
        x1 = torch.cat([anchor,positive],dim=2)
        
        rep_anchor = anchor.repeat(negative.shape[0],1,1)
        x2 = torch.cat([rep_anchor,negative],dim=2)
        
        x3 = torch.cat([x1,x2],dim=0)
        pred = self.last_layer(self.fc(x3)).squeeze()
        #nrom_pred = F.log_softmax(pred,dim=1)
        target = torch.ones(pred.shape[0],dtype=torch.float32).to(pred.device)
        target[1:]=0.0
        #target = target.float32()
        return self.loss(pred,target)
        
    def __str__(self):
        return f"pcl_binary_loss"
    