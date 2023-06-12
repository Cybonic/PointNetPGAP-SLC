
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
import numpy as np
from utils.eval import eval_place

def test_eval(queries,descriptors,poses):
    k_top_cands = 25
    metrics = ["tp","RR","recall","MRR"]
    out,predictions = eval_place(queries,descriptors,poses,k=k_top_cands)
    assert isinstance(out,dict)
    assert all([True for m in metrics if m in out])
    print("[PASSED] RANKING METRICS")
    
    loop_cand=predictions['loop_cand']
    loop_scores=predictions['loop_scores']
    gt_loops = predictions['gt_loops']

    assert isinstance(predictions['loop_cand'],np.ndarray)
    assert loop_cand.shape[0] == queries.shape[0]
    assert loop_cand.shape[1] == k_top_cands
    print("[PASSED] LOOP CANDIDATES")
    
    assert isinstance(loop_scores,np.ndarray)
    assert loop_scores.shape[0] == queries.shape[0]
    assert loop_scores.shape[1] == k_top_cands
    print("[PASSED] LOOP SCORES")

    assert isinstance(gt_loops,np.ndarray)
    assert gt_loops.shape[0] == queries.shape[0]
    assert gt_loops.shape[1] == k_top_cands
    print("[PASSED] LOOP SCORES")
    
    

def test_rerank_eval(queries,descriptors,poses,rerank_idx):
    metrics = ["tp","RR","recall","MRR",
               "tp_rr","RR_rr","recall_rr","MRR_rr"]
    k_top_cands=25
    out,predictions = eval_place(queries,descriptors,poses,
                                    k=k_top_cands,reranking=rerank_idx)
    
    assert isinstance(out,dict)
    assert all([True for m in metrics if m in out])
    print("[PASSED] RE-RANKING METRICS")

    loop_candidates=predictions['loop_cand']
    loop_scores=predictions['loop_scores']
    gt_loops = predictions['gt_loops']

    assert isinstance(loop_candidates,np.ndarray)
    assert loop_candidates.shape[0] == queries.shape[0]
    assert loop_candidates.shape[1] == k_top_cands
    print("[PASSED] RE-RANKING LOOP CANDIDATES")
    
    assert isinstance(loop_scores,np.ndarray)
    assert loop_scores.shape[0] == queries.shape[0]
    assert loop_scores.shape[1] == k_top_cands
    print("[PASSED] RE-RANKING LOOP SCORES")

    assert isinstance(gt_loops,np.ndarray)
    assert gt_loops.shape[0] == queries.shape[0]
    assert gt_loops.shape[1] == k_top_cands
    print("[PASSED] LOOP SCORES")
    

def run_test_eval(n_samples,dim,n_queries):
    
    descriptors = np.random.random((n_samples,dim))
    print(descriptors.shape)
    poses = np.random.random((n_samples,3))
    print(poses.shape)
    queries = np.sort(np.random.randint(50,100,n_queries))
    print(queries.shape)
    test_eval(queries,descriptors,poses)
    re_ranked_idx = np.random.randint(0,25,(n_queries,25))
    #print(f"Reranking shape {re_ranked_idx.shape}")
    test_rerank_eval(queries,descriptors,poses,re_ranked_idx)
   
    
    

if __name__=="__main__":

    root = "/home/tiago/Dropbox/research/datasets"
    print("\n************************************************")
    print("Kitti Eval Test\n")
    run_test_eval(100,256,10)

    print("************************************************\n")