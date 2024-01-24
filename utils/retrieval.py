
import numpy as np


def comp_queries_score_table(target,queries):
    '''
    
    '''
    if not isinstance(target,np.ndarray):
        target = np.array(target)
    
    if not isinstance(queries,np.ndarray):
        queries = np.array(queries)

    table_width = target.shape[0]
    idx = np.arange(table_width)
    target_wout_queries = np.setxor1d(idx,queries)
    
    table = np.zeros((queries.shape[0],target_wout_queries.shape[0]),dtype=np.float32)
    
    for i,(q) in enumerate(queries):
        qdistance = np.linalg.norm(target[q,:]-target[target_wout_queries,:],axis=1)
        table[i,:]= qdistance
    
    table_struct = {'idx':target_wout_queries,'table':table}
    return(table_struct)



def comp_score_table(target):
    '''
    
    '''
    if not isinstance(target,np.ndarray):
        target = np.array(target)
    
    table_width = target.shape[0]
    
    table = np.zeros((table_width,table_width),dtype=np.float32)
    table = []
    for i in range(table_width):
        qdistance = np.linalg.norm(target[i,:]-target,axis=1)
        table.append(qdistance.tolist())
    return(np.asarray(table))




def gen_ground_truth(pose,anchor,pos_thres,neg_thres,num_neg,num_pos):
    '''
    input args: 
        pose [nx3] (x,y,z)
        anchor [m] indices
        pos_thres (scalar): max range of positive samples 
        neg_thres (scalar): min range of negative samples 
        num_neg (scalar): number of negative sample to return
        num_pos (scalar): number of positive sample to return
    
    return 
        positive indices wrt poses
        negative indices wrt poses
    '''
    # assert mode in ['hard','distribution']

    table  = comp_score_table(pose)

    all_idx = np.arange(table.shape[0])
    wout_query_idx = np.setxor1d(all_idx,anchor)
    positive = []
    negative = []
    
    for a in zip(anchor):

        query_dist = table[a]
        #selected_idx = np.where(query_dist>0)[0] # exclude anchor idx (dist = 0)
        #sort_query_dist_idx  =  np.argsort(query_dist)
        all_pos_idx  = np.where(query_dist < pos_thres)[0]
        sort_pos_idx = np.argsort(query_dist[all_pos_idx])
        sort_all_pos_idx = all_pos_idx[sort_pos_idx]
        
        dis = query_dist[sort_all_pos_idx]
        all_pos_idx  = sort_all_pos_idx[1:] # remove the 0 element 
     
        dis_top = query_dist[all_pos_idx]

        tp  = np.array([i for i in all_pos_idx if i not in anchor])
        #tp = np.setxor1d(all_pos_idx,anchor)
        if len(tp)>num_pos and num_pos>0:
            tp = tp[:num_pos]
            dis_top = query_dist[tp]
            #pos_idx = np.random.randint(low=0,high = len(tp),size=num_pos)
            #tp = tp[pos_idx]

        all_neg_idx = np.where(query_dist>neg_thres)[0]
        neg_idx = np.random.randint(low=0,high = len(all_neg_idx),size=num_neg)
        tn = all_neg_idx[neg_idx]
        
        positive.append(tp)
        negative.append(tn)

    return(np.array(positive),np.array(negative))



def comp_gt_table(pose,anchors,pos_thres):
    '''
    
    
    '''
    table  = comp_score_table(pose)
    num_pose = pose.shape[0]
    gt_table = np.zeros((num_pose,num_pose),dtype=np.uint8)
    all_idx  = np.arange(table.shape[0])
    idx_wout_anchors = np.setxor1d(all_idx,anchors) # map idx: ie all idx excep anchors

    for anchor in anchors:
        anchor_dist = table[anchor]
        all_pos_idx = np.where(anchor_dist < pos_thres)[0] # Get all idx on the map that form a loop (ie dist < thresh)
        tp = np.intersect1d(idx_wout_anchors,all_pos_idx).astype(np.uint32) # Remove those indices that belong to the anchor set
        gt_table[anchor,tp] = 1 # populate the table with the true positives

    return(gt_table)




def evaluation(relevant_hat,true_relevant, top=1,mode = 'relaxe'):
    '''
    https://amitness.com/2020/08/information-retrieval-evaluation/
    
    '''
    return  
    
    #return 



def evaluationv2(pred,gt,type = 'hard',smooth=0.000001):
    '''
    https://amitness.com/2020/08/information-retrieval-evaluation/
    
    '''
    P = np.sum(gt == 1)
    N = np.sum(gt == 0)

    num_samples = pred.shape[0]
    
    eval_rate = []
    for pred_line,gt_line in zip(pred,gt):
        tp,fp,tn,fn = 0,0,0,0
        if np.sum(gt_line == 1)>0:
            tp = 1 if np.sum(((gt_line == 1) & (pred_line==1))==1) > 0 else 0
            if tp ==0:
                fn = 1 if np.sum(((gt_line == 1) & (pred_line==0))==1) > 0 else 0
        else:
            positives = np.sum(pred_line == 1)
            fp = 1 if positives > 0  else 0
            tn = 1 if positives == 0 else 0
        eval_rate.append([tp,fp,tn,fn])

    rate = np.sum(eval_rate,axis=0)
    tp,fp,tn,fn = rate[0],rate[1],rate[2],rate[3]

    return tp,fp,tn,fn 



def retrieval_metric(tp,fp,tn,fn):

    b = tp+tn+fp+fn  
    accuracy = (tp+tn)/b if b!=0 else 0

    b = tp+fn
    recall = tp/b if b!=0 else 0

    b = tp+fp
    precision = tp/b if b!=0 else 0
       
    b = precision + recall 
    F1 = 2*((precision*recall) /(precision + recall)) if b!=0 else 0

    return({'recall':recall, 'precision':precision,'F1':F1,'accuracy':accuracy})
