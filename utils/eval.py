
import numpy as np
import torch
import os
import tqdm

def save_results_csv2(self,file,results,top,**argv):
    import pandas as pd
    if file == None:
      raise NameError    #file = self.results_file # Internal File name 
    
    decimal_res = 3
    metrics = ['recall','recall_rr','MRR','MRR_rr','mean_t_RR'] # list(results.keys())[5:]
    recall= np.round(np.transpose(np.array(results['recall'][25])),decimal_res).reshape(-1,1)
    recall_rr= np.round(np.transpose(np.array(results['recall_rr'][25])),decimal_res).reshape(-1,1)

    scores = np.zeros((recall.shape[0],3))
    scores[0,]
    scores[0,0]= np.round(results['MRR'][25],decimal_res)
    scores[0,1]= np.round(results['MRR_rr'][25],decimal_res)
    scores[0,2]= results['mean_t_RR']
    scores = np.concatenate((recall,recall_rr,scores),axis=1)

    df = pd.DataFrame(scores,columns = metrics)
    best = np.round(results['recall_rr'][25][top-1],decimal_res)
    checkpoint_dir = ''
    filename = os.path.join(checkpoint_dir,f'{file}-{str(best)}.csv')
    df.to_csv(filename)




def eval_place(queries,descriptrs,poses,k=25,radius=[25],reranking = None,window=1):
  """_summary_

  Args:
      queries (numpy): query indices 
      descriptrs (numpy): array with all descriptors [n x d], 
                          where n is the number of descriptors and 
                          ´d´ is the dimensionality;
      poses (numpy): array with all the poses [n x 3], where
                    n is the number os positions; 
      k (int, optional): number of top candidates. Defaults to 25.
      radius (list, optional): radius in meters [m] of true loops. Defaults to [25].
      reranking (numpy, optional): _description_. Defaults to None.
      window (int, optional): _description_. Defaults to 1.

  Returns:
      dict: retrieval metrics,
      numpy int: loop candidates 
      numpy float: loop scores
  """
  if not isinstance(queries,np.ndarray):
     queries = np.array(queries)
     
  n_frames = queries.shape[0]
  if isinstance(descriptrs,dict):
    descriptrs = np.array(list(descriptrs.values()))
  #else:
  map_indices = np.arange(descriptrs.shape[0])
  
  # Initiate evaluation dictionary  
  global_metrics = {'tp': {r: [0] * k for r in radius}}
  global_metrics['RR'] = {r: [] for r in radius}
  
  # Initiate evaluation dictionary for re-ranking
  if isinstance(reranking,(np.ndarray, np.generic,list)):
    global_metrics['RR_rr'] = {r: [] for r in radius}
    global_metrics['t_RR'] = []
    global_metrics['tp_rr'] = {r: [0] * k for r in radius}
  
  loop_cands = []
  loop_scores= []
  gt_loops   = []
  for i,(q) in enumerate(queries):
    
    query_pos = poses[q]
    query_destps = descriptrs[q]

    #q = queries[query_ndx]
    map_idx = np.arange(q-window) # generate indices until q - window
    filtered_map_idx = map_indices[map_idx]
    selected_poses = poses[filtered_map_idx]
    selected_desptrs = descriptrs[filtered_map_idx]
    
    # Compute loop candidates
    delta_dscpts = query_destps - selected_desptrs
    embed_dist = np.linalg.norm(delta_dscpts, axis=-1)
    nn_ndx = np.argsort(embed_dist)[:k]
    embed_dist = embed_dist[nn_ndx]
    
    # compute ground truth distance
    delta = query_pos - selected_poses
    euclid_dist = np.linalg.norm(delta, axis=-1)
    gt_loops.append(np.argsort(euclid_dist)[:k])

    euclid_dist_top = euclid_dist[nn_ndx]
    

    # Count true positives for different radius and NN number
    global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist_top[:nn + 1] <= r).any() 
                                                                  else 0) for nn in range(k)] for r in radius}
    global_metrics['RR'] = {r: global_metrics['RR'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist <= r) if x), 0)]
                                                                  for r in radius}

    if isinstance(reranking,(np.ndarray, np.generic,list)):
      nn_ndx = nn_ndx[reranking[i]]
      embed_dist = embed_dist[reranking[i]]
      euclid_dist_rr = euclid_dist[nn_ndx]
      global_metrics['tp_rr'] = {r: [global_metrics['tp_rr'][r][nn] + (1 if (euclid_dist_rr[:nn + 1] <= r).any() else 0) for nn in range(k)] for r in radius}
      global_metrics['RR_rr'] = {r: global_metrics['RR_rr'][r]+[next((1.0/(i+1) for i, x in enumerate(euclid_dist_rr <= r) if x), 0)] for r in radius}
    
    # save loop candidates indices 
    loop_cands.append(nn_ndx)
    loop_scores.append(embed_dist)
    
  # Calculate mean metrics
  global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / n_frames for nn in range(k)] for r in radius}
  global_metrics['MRR'] = {r: np.mean(np.asarray(global_metrics['RR'][r])) for r in radius}
  
  if isinstance(reranking,(np.ndarray, np.generic,list)):
    global_metrics["recall_rr"] = {r: [global_metrics['tp_rr'][r][nn] / n_frames for nn in range(k)] for r in radius}
    global_metrics['MRR_rr'] = {r: np.mean(np.asarray(global_metrics['RR_rr'][r])) for r in radius}
  
  #global_metrics['mean_t_RR'] = np.mean(global_metrics['t_RR'])
  prediction =  {'loop_cand':np.array(loop_cands),
                 'loop_scores':np.array(loop_scores),
                 'gt_loops':np.array(gt_loops)}
  return global_metrics,prediction




def eval_row_relocalization(descriptrs,poses, row_labels, n_top_cand=25, radius=[25],window=1,warmup = 100, sim='L2'):
  """_summary_

  Args:
      queries (numpy): query indices 
      descriptrs (numpy): array with all descriptors [n x d], 
                          where n is the number of descriptors and 
                          ´d´ is the dimensionality;
      poses (numpy): array with all the poses [n x 3], where
                    n is the number os positions; 
      k (int, optional): number of top candidates. Defaults to 25.
      radius (list, optional): radius in meters [m] of true loops. Defaults to [25].
      reranking (numpy, optional): _description_. Defaults to None.
      window (int, optional): _description_. Defaults to 1.

  Returns:
      dict: retrieval metrics,
      numpy int: loop candidates 
      numpy float: loop scores
  """
  assert isinstance(n_top_cand,int), "Number of top candidates must be an integer"  
  assert n_top_cand > 0, "Number of top candidates must be greater than 0"

  assert isinstance(radius,list), "Radius must be a list"
     

  if isinstance(descriptrs,dict):
    descriptrs = np.array(list(descriptrs.values()))

  # Normalize descriptors
  descriptrs = descriptrs / np.linalg.norm(descriptrs, axis=-1, keepdims=True) 
  descriptrs = (descriptrs + 1)/2
  assert np.sum(descriptrs,axis=-1).all() == 1, "Descriptors must be normalized"
  
  #descriptrs = np.linalg.norm(descriptrs, axis=-1) 
  all_indices = np.arange(descriptrs.shape[0])
  from utils.metric import reloc_metrics
  
  metric = reloc_metrics(n_top_cand,radius)

  ignore_indices = warmup
  if ignore_indices < window:
    ignore_indices = window + 10
    
  queries = all_indices[ignore_indices:]

  gt_loops   = []
  
  predictions = {}
  
  import tqdm
  poses[:,2] = 0 # ignore z axis
  for i,(q) in tqdm.tqdm(enumerate(queries),total=len(queries)):
    
    query_pos = poses[q,:]
    query_destps = descriptrs[q]
    query_labels = row_labels[q]
    
    # Ignore scans within a window around the query
    selected_map_idx = np.arange(0,q-window) # generate indices until q - window 
  
    selected_poses   = poses[selected_map_idx,:]
    selected_desptrs = descriptrs[selected_map_idx,:]
    selected_map_labels   = row_labels[selected_map_idx]
    # ====================================================== 
     # compute ground truth distance
    delta = query_pos.reshape(1,3) - selected_poses
    gt_euclid_dist = np.linalg.norm(delta, axis=-1) 
    
    # TRUE LOOPS
    true_loops_idx = np.argsort(gt_euclid_dist)[:n_top_cand]
    # Get labels of true loops
    true_loops_labels = selected_map_labels[true_loops_idx]
    true_loop_same_row_bool = query_labels == true_loops_labels
    # filter true loops that are not in the same row
    true_loops_inrow_idx = true_loops_idx[true_loop_same_row_bool]
    true_loops_inrow_dist = gt_euclid_dist[true_loops_inrow_idx]
    
    gt_loops.append(true_loops_inrow_idx)
    
     # CANDIDATES LOOP
    #delta_dscpts = np.dot(query_destps,selected_desptrs.transpose())
    delta_dscpts = query_destps - selected_desptrs
    
    embed_sim = np.linalg.norm(delta_dscpts, axis=-1) # Euclidean distance

    # Sort to get the most similar (lowest values) vectors first
    cand_loop_idx = np.argsort(embed_sim)[:n_top_cand]
    cand_dist = gt_euclid_dist[cand_loop_idx]
    cand_sim  = embed_sim[cand_loop_idx]
    cand_labels = selected_map_labels[cand_loop_idx]
    
    # filter  same row
    cand_inrow_bool  = query_labels == cand_labels
    
    metric.update(true_loops_inrow_dist,cand_dist,cand_sim,cand_inrow_bool)
    # return the euclidean distance of the top descriptor predictions
    
    predictions[q]={'label':query_labels,
     'true_loops':{'idx':true_loops_inrow_idx,'dist':true_loops_inrow_dist,'labels':true_loops_labels},
      'cand_loops':{'idx':cand_loop_idx,'dist':cand_dist,'sim':cand_sim,'labels':cand_labels}}
    


    # save loop candidates indices 
    
  
  global_metrics = metric.get_metrics()
  #global_metrics['mean_t_RR'] = np.mean(global_metrics['t_RR'])
  #prediction =  {'loop_cand':loop_cands,
  #               'loop_scores':loop_scores,
  #               'gt_loops':gt_loops}
  
  return global_metrics,predictions



def eval_row_place(queries,descriptrs,poses, row_labels, n_top_cand=25,radius=[25],window=1,sim = 'L2'):
  """_summary_

  Args:
      queries (numpy): query indices 
      descriptrs (numpy): array with all descriptors [n x d], 
                          where n is the number of descriptors and 
                          ´d´ is the dimensionality;
      poses (numpy): array with all the poses [n x 3], where
                    n is the number os positions; 
      k (int, optional): number of top candidates. Defaults to 25.
      radius (list, optional): radius in meters [m] of true loops. Defaults to [25].
      reranking (numpy, optional): _description_. Defaults to None.
      window (int, optional): _description_. Defaults to 1.

  Returns:
      dict: retrieval metrics,
      numpy int: loop candidates 
      numpy float: loop scores
  """
  assert isinstance(n_top_cand,int), "Number of top candidates must be an integer"  
  assert n_top_cand > 0, "Number of top candidates must be greater than 0"

  assert isinstance(radius,list), "Radius must be a list"

  if not isinstance(queries,np.ndarray):
     queries = np.array(queries)
     

  if isinstance(descriptrs,dict):
    descriptrs = np.array(list(descriptrs.values()))

  all_map_indices = np.arange(descriptrs.shape[0])
  from utils.metric import retrieval_metrics
  metric = retrieval_metrics(n_top_cand,radius)
  
  loop_cands = []
  loop_scores= []
  gt_loops   = []
  
  poses[:,2] = 0 # ignore z axis
  for i,(q) in tqdm.tqdm(enumerate(queries),total = len(queries),desc='Evaluating Retrieval'):
    
    query_pos = poses[q,:]
    query_destps = descriptrs[q]
    query_label = row_labels[q]
    
    # Ignore scans within a window around the query
    q_map_idx = np.arange(0,q-window,dtype=np.uint32) # generate indices until q - window 
    selected_map_idx = all_map_indices[q_map_idx]

    selected_poses   = poses[selected_map_idx,:]
    selected_desptrs = descriptrs[selected_map_idx,:]
    selected_map_labels   = row_labels[selected_map_idx]
    # ====================================================== 
    # compute ground truth distance
    
    delta = query_pos.reshape(1,3) - selected_poses
    gt_euclid_dist = np.linalg.norm(delta, axis=-1)
       
    # return the indices of the sorted array
    gt_loops.append(np.argsort(gt_euclid_dist)[:n_top_cand])
    
    
    
     # Compute loop candidates
    if sim == 'L2':
      delta_dscpts = query_destps - selected_desptrs
      embed_dist = np.linalg.norm(delta_dscpts, axis=-1) # Euclidean distance
    elif sim == 'sc_similarity':
      import networks.scancontext.scancontext as sc
      embed_dist = sc.sc_dist(query_destps,selected_desptrs)
    else:
      raise NameError('Similarity metric not implemented')
    
    
    
    # Sort to get the most similar (lowest values) vectors first
    est_loop_cand_idx = np.argsort(embed_dist)#[:n_top_cand]
    
    est_loop_cand_sim_dist = embed_dist[est_loop_cand_idx]
    gt_loops_cand_euclid_dist = gt_euclid_dist[est_loop_cand_idx]
    loop_cand_labels = selected_map_labels[est_loop_cand_idx]
    #cand_in_same_row_idx = np.where(query_labels == loop_cand_labels)[0]
    cand_in_same_row_bool = query_label == loop_cand_labels
    metric.update(gt_loops_cand_euclid_dist,cand_in_same_row_bool)

    # save loop candidates indices 
    loop_cands.append(est_loop_cand_idx)
    loop_scores.append(est_loop_cand_sim_dist)
  
  global_metrics = metric.get_metrics()
  #global_metrics['mean_t_RR'] = np.mean(global_metrics['t_RR'])
  prediction =  {'loop_cand':loop_cands,
                 'loop_scores':loop_scores,
                 'gt_loops':gt_loops}
  
  return global_metrics,prediction


def comp_pair_permutations(n_samples):
    combo_idx = torch.arange(n_samples)
    permutation = torch.from_numpy(np.array([np.array([a, b]) for idx, a in enumerate(combo_idx) for b in combo_idx[idx + 1:]]))
    return permutation[:,0],permutation[:,1]



def comp_loops(sim_map,queries,window=500,max_top_cand=25):
  loop_cand = []
  loop_sim = []
  #eu_value = np.linalg.norm(x - data,axis=1)
  for i,q in enumerate(queries):
    sim = sim_map[i] # get loop similarities for query i 
    bottom = q-window # 
    elegible = sim[:bottom] 
    #elegible = sim
    cand = np.argsort(elegible)[:max_top_cand] # sort similarities and get top N candidates
    sim = elegible[cand]
    loop_sim.append(sim)
    loop_cand.append(cand)
  return np.array(loop_cand), np.array(loop_sim)




def calculateMahalanobis(y=None, data=None, inv_covmat=None):
  
    y_mu = y - data
    #if not cov:
    #    cov = np.cov(data.values.T)
    #inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    #return np.sqrt(mahal)
    return  np.sqrt(mahal.diagonal())

def eval_row_retrieval(queries,descriptrs,poses, row_labels, n_top_cand=25,radius=[25],window=1):
   
  if not isinstance(queries,np.ndarray):
     queries = np.array(queries)
     
  n_frames = queries.shape[0]
  if isinstance(descriptrs,dict):
    descriptrs = np.array(list(descriptrs.values()))
  #else:
  all_map_indices = np.arange(descriptrs.shape[0])
  
  # Initiate evaluation dictionary  
  global_metrics = {'tp': {r: [0] * n_top_cand for r in radius}}
  global_metrics['RR'] = {r: [] for r in radius}



def retrieve_eval(retrieved_map,true_relevant_map,top=1,**argv):
  '''
  In a relaxed setting, at each query it is only required to retrieve one loop. 
  so: 
    Among the retrieved loop in true loop 
    recall  = tp/1
  '''
  assert top > 0
  n_queries = retrieved_map.shape[0]
  precision, recall = 0,0
  for retrieved,relevant in zip(retrieved_map,true_relevant_map):
    top_retrieved = retrieved[:top] # retrieved frames for a given query
    
    tp = 0 # Reset 
    if any(([True  if cand in relevant else False for cand in top_retrieved])):
        # It is only required to find one loop per anchor in a set of retrieved frames
        tp = 1 
    
    recall += tp # recall = tp/1
    precision += tp/top # retrieved loops/ # retrieved frames (top candidates) (precision w.r.t the query)
  
  recall /= n_queries  # average recall of all queries 
  precision /= n_queries  # average precision of all queries 

  return({'recall':recall,'precision':precision})