
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



def eval_row_place(queries,descriptrs,poses,segment_labels, n_top_cand=25,radius=[25],window=1,sim = 'L2'):
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
    descriptrs = np.array([d['d'] for d in descriptrs.values()])
  #descriptrs = np.array(list(descriptrs.values()))

  all_map_indices = np.arange(descriptrs.shape[0])
  from utils.metric import retrieval_metrics
  
  # Count number -1 in segment labels
  
  count = np.sum(segment_labels == -1)
  print(f'Number of -1 in segment labels: {count}')
  max_segments = np.max(segment_labels)
  metric = retrieval_metrics(n_top_cand,radius,n_segments=max_segments+1)

  
  to_store= {}
  
  poses[:,2] = 0 # ignore z axis
  for i,(q) in tqdm.tqdm(enumerate(queries),total = len(queries),desc='Evaluating Retrieval'):
    
    query_pos = poses[q,:]
    query_destps = descriptrs[q]
    query_label = segment_labels[q]
    
    # Ignore scans within a window around the query
    q_map_idx = np.arange(0,q-window,dtype=np.uint32) # generate indices until q - window 
    selected_map_idx = all_map_indices[q_map_idx]

    selected_poses   = poses[selected_map_idx,:]
    selected_desptrs = descriptrs[selected_map_idx,:]
    selected_map_labels   = segment_labels[selected_map_idx]
    
    # ====================================================== 
    # compute ground truth distance
    delta = query_pos.reshape(1,3) - selected_poses
    gt_euclid_dist = np.linalg.norm(delta, axis=-1)
    
    gt_loop_idx = np.argsort(gt_euclid_dist)[:n_top_cand]
    gt_loop_L2 = gt_euclid_dist[gt_loop_idx]
    gt_loop_labels = selected_map_labels[gt_loop_idx]

    
     # Compute loop candidates
    if sim == 'L2':
      delta_dscpts = query_destps - selected_desptrs
      embed_dist = np.linalg.norm(delta_dscpts, axis=-1) # Euclidean distance
    elif sim == 'cosine':
      import utils.loss as loss
      embed_dist =loss.cosine_torch_loss(query_destps,selected_desptrs,dim=1).cpu().numpy()
    else:
      raise NameError('Similarity metric not implemented')
    
    
    # Sort to get the most similar (lowest values) vectors first
    est_loop_cand_idx = np.argsort(embed_dist)[:n_top_cand]
    
    pred_loop_dist = embed_dist[est_loop_cand_idx]
    pred_loop_L2 = gt_euclid_dist[est_loop_cand_idx]
    pred_loop_labels = selected_map_labels[est_loop_cand_idx]

    if not (query_label == gt_loop_labels).any():
      # Guarantee that there exists ground truth loops in the same segment
      # as the query
      continue
    
    metric.update(query_label,pred_loop_labels,pred_loop_L2,gt_loop_L2)

    # save loop candidates indices 
    to_store[q]= {'segment':query_label,'true_loops':{'idx':gt_loop_idx,'dist':gt_loop_L2,'segment':gt_loop_labels},
                                        'pred_loops':{'idx':est_loop_cand_idx,'dist':pred_loop_dist,'segment':pred_loop_labels}}
  
  global_metrics = metric.get_metrics()
 
 
  return global_metrics,to_store

