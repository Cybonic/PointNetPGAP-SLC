

import numpy as np


class reloc_metrics:
  '''
  This class is used to compute the metrics for the retrieval task
  '''
  def __init__(self,k_top = 10, radius = [1,2,3,4,5,6,7,8,9,10],sim_thresh = 0.5):
    self.k_top = k_top
    self.radius = radius
    self.sim_thresh = sim_thresh
    self.reset()
    
  
  def reset(self):
    self.n_updates = 0
    self.global_metrics = {'tp': {r: [0]*(self.k_top+1) for r in self.radius},
                                'fp': {r: [0]*(self.k_top+1) for r in self.radius},
                                'fn': {r: [0]*(self.k_top+1) for r in self.radius},
                                'tn': {r: [0]*(self.k_top+1) for r in self.radius},
                                'RR': {r: [] for r in self.radius},
                                'precision': {r: [0]*(self.k_top+1) for r in self.radius},
                                'recall': {r: [0]*(self.k_top+1) for r in self.radius},
    }
    
    
  def update(self,true_loop_dist,cand_dist,similarity,is_inrow):
    
    # Update global metrics
    correct = {r: [0]*(self.k_top+1) for r in self.radius}
    wrong = {r: [0]*(self.k_top+1) for r in self.radius}
    
    self.n_updates += 1
    for r in self.radius:
      for nn in range(0,self.k_top):
        
        num_loop = len(np.where(true_loop_dist[:nn+1]<r)[0] )
        
        same_row = is_inrow[:nn+1]
        cand_range = cand_dist[:nn + 1]
        sim = similarity[:nn+1]
        
        if num_loop > 0 and same_row.any():
          if (cand_range <= r).any() and (sim <= self.sim_thresh).any():
            self.global_metrics['tp'][r][nn] += 1 
            #correct[r][nn] = 1
          if (cand_range <= r).any() and (sim > self.sim_thresh).any():
            self.global_metrics['fn'][r][nn] += 1
            #wrong[r][nn] = 1
          if (cand_range >  r).any() and (sim<= self.sim_thresh).any():
            self.global_metrics['fp'][r][nn] += 1
            #wrong[r][nn] = 1
          if (cand_range >  r).any() and (sim > self.sim_thresh).any():
            self.global_metrics['tn'][r][nn] += 1
          
        else:
          self.global_metrics['tn'][r][nn] += 1 if len(cand_range) ==0 and len(sim)==0 else 0
          self.global_metrics['tn'][r][nn] += 1 if (cand_range > r).any() and (sim > self.sim_thresh).any() else 0
          self.global_metrics['fp'][r][nn] += 1 if (cand_range > r).any() and (sim <= self.sim_thresh).any() else 0
          #self.global_metrics['nl']['fp'][r][nn] += 1 if (can <= r).any() and (sim > self.sim_thresh).any() else 0
          #self.global_metrics['nl']['tn'][r][nn] += 1 if (can <= r).any() and (sim<= self.sim_thresh).any() else 0
          
  
        self.global_metrics['recall'][r][nn] = (self.global_metrics['tp'][r][nn] + self.global_metrics['fn'][r][nn])/self.n_updates
        self.global_metrics['precision'][r][nn] = ((self.global_metrics['tp'][r][nn] + self.global_metrics['fn'][r][nn])/(nn+1))/self.n_updates
        
    return {'recall':self.global_metrics['recall'] , 'precision':self.global_metrics['precision']}
  
  def get_metrics(self):
    return self.global_metrics



class retrieval_metrics:
  '''
  This class is used to compute the metrics for the retrieval task
  '''
  def __init__(self,k_top = 10, radius = [1,2,3,4,5,6,7,8,9,10]):
    self.k_top = k_top
    self.radius = radius
    self.reset()
    
  
  def reset(self):
    self.n_updates = 0
    self.global_metrics = {'tp': {r: [0]*(self.k_top+1) for r in self.radius},
                      'RR': {r: [] for r in self.radius},
                      'precision': {r: [0]*(self.k_top+1) for r in self.radius},
                      'recall': {r: [0]*(self.k_top+1) for r in self.radius},
    }
    
  def update(self,cand_true_dist,cand_in_row):
    
    # Update global metrics
    self.n_updates += 1
    for r in self.radius:
      for nn in range(0,self.k_top):
        # Get the top-k candidates
        label = cand_in_row[:nn + 1]
        dist  = cand_true_dist[:nn + 1]
        # Verify if there is a loop in the top-k candidates
        cand_in_range = np.where(dist <= r)[0]
        # Verify if the loop is in the same row
        cand_inrange_and_inrow = label[cand_in_range]
        # Update metrics
        if (cand_inrange_and_inrow == True).any():
          self.global_metrics['tp'][r][nn] += 1
          
        self.global_metrics['recall'][r][nn] = self.global_metrics['tp'][r][nn]/self.n_updates
        self.global_metrics['precision'][r][nn] = (self.global_metrics['tp'][r][nn]/(nn+1))/self.n_updates
      
    return {'recall':self.global_metrics['recall'] , 'precision':self.global_metrics['precision']}
  
  def get_metrics(self):
    return self.global_metrics
    
def relocal_metric(relevant_hat,true_relevant):
    '''
    Difference between relocal metric and retrieval metric is that 
    retrieval proseposes that only evalautes positive queries
    ...

    input: 
    - relevant_hat (p^): indices of 
    '''
    n_samples = len(relevant_hat)
    recall,precision = 0,0
    
    for p,g in zip(relevant_hat,true_relevant):
        p = np.array(p).tolist()
        n_true_pos = len(g)
        n_pos_hat = len(p)
        tp = 0 
        fp = 0
        if n_true_pos > 0: # postive 
            # Loops exist, we want to know if it can retireve the correct frame
            num_tp = np.sum([1 for c in p if c in g])
            
            if num_tp>0: # found at least one loop 
                tp=1
            else:
                fn=1 
            
        else: # Negative
            # Loop does not exist: we want to know if it can retrieve a frame 
            # with a similarity > thresh
            if n_pos_hat == 0:
                tp=1

        recall += tp/1 
        precision += tp/n_pos_hat if n_pos_hat > 0 else 0
    
    recall/=n_samples
    precision/=n_samples
    return {'recall':recall, 'precision':precision}


def retrieve_eval(retrieved_map,true_relevant_map,top=1,**argv):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert top > 0
  #assert true_relevant_map is bool 
  #assert isinstance(true_relevant_map,
  n_queries = len(retrieved_map)
  precision, recall = 0,0
  for retrieved,loops_bools in zip(retrieved_map,true_relevant_map):
    top_retrieved = retrieved[:top] # retrieved frames for a given query
    pred_retrieval = loops_bools[retrieved]
    tp = 0 # Reset 
    #if any(([True  if cand in relevant else False for cand in top_retrieved])):
    if any(pred_retrieval):    
        # It is only required to find one loop per anchor in a set of retrieved frames
        tp = 1 
    
    recall += tp # recall = tp/1
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r.t the query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})




def retrieve_metricsv2(relevant_hat,true_relevant,top=1,mode = 'hard'):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert mode in ['relaxe', 'hard']

  n_queries = relevant_hat.shape[0]
  precision, recall = 0,0
  for p,g in zip(relevant_hat,true_relevant):
    top_relevant_hat = p[:top]

    n_true_loops = 1 if mode =='relaxe' else len(g) # Number of loops per anchor = 1
    
    num_tp = np.sum([1 for c in top_relevant_hat if c in g]) # Number of predicted loops
    tp  = 1 if mode == 'relaxe' and num_tp > 0 else num_tp  # In "relaxe" mode, 
    
    recall +=tp/n_true_loops #  (Recall w.r. to query) retrieved loops/ universe of loops 
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r. to query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})