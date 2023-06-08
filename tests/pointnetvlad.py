
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from tests.laserscan import test_laserscan_tensor

from networks.network_pipeline import get_pipeline
from PIL import Image
import numpy as np
import torch

def test_PointNetVLAD(input,model_name:str,out_dim:int):
    '''
    This test checks if the overlap transformer outputs the correct descriptor size
    Input parameters:
        input (np.ndarray) input data  
        model_name (str)  model name 
        out_dim (int) descriptor size 
    
    Output:
     - Test (Bool)
     - generated descriptor (torch tensor)
    '''
    assert torch.is_tensor(input),"data should be from the SparseTensor type"
    pcl_sparse = input.cuda()
    
    batch = torch.stack((pcl_sparse,pcl_sparse),dim=0)
    # Get Model
    model = get_pipeline(model_name,output_dim=out_dim)
    model = model.cuda()
    assert str(model) == model_name,'Name does not match'
    # Prediction
    descriptor,features = model(batch)
    assert torch.is_tensor(descriptor), "output should be a tensor"
    b,d = descriptor.shape
    assert b == 2,'Batch size is wrong'
    assert d == out_dim, "descriptor shape is wrong"

    return True, descriptor

def run_PointNetVLAD_test(file):
    flag,sparse_pcl = test_laserscan_tensor(file)
    flag,dptrs = test_PointNetVLAD(sparse_pcl,'PointNetVLAD',out_dim = 256)
    print("[PASSED] PointNetVLAD")


if __name__ == "__main__":

    from networks.pipelines.pipeline_utils import *
    file = 'tutorial_data/000000.bin'
    run_PointNetVLAD_test(file)
    
