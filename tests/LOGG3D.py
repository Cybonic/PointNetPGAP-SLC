
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from tests.sparselaserscan import test_sparse_tensor
from dataloader.sparselaserscan import SparseLaserScan
from networks.network_pipeline import get_pipeline
from PIL import Image
import numpy as np
import torch
from torchsparse import SparseTensor
from torchsparse.utils.collate import sparse_collate

def test_LOGG3D(input,model_name:str,out_dim:int):
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
    assert isinstance(input,SparseTensor), "data should be from the SparseTensor type"
    pcl_sparse = input.cuda()
    
    batch = [pcl_sparse,pcl_sparse]
    # Create Batch
    pcl_sparse = sparse_collate(batch).cuda()
    pcl_sparse.C = pcl_sparse.C.int()
    # Get Model
    model = get_pipeline(model_name,output_dim=out_dim)
    model = model.cuda()
    assert str(model) == model_name,'Name does not match'
    # Prediction
    descriptor,features = model(pcl_sparse)
    assert torch.is_tensor(descriptor), "output should be a tensor"
    b,d = descriptor.shape
    assert b == 2,'Batch size is wrong'
    assert d == out_dim, "descriptor shape is wrong"

    return True, descriptor

def run_LOGG3D_test(file,voxel):
    flag,sparse_pcl = test_sparse_tensor(file,voxel)
    flag,dptrs = test_LOGG3D(sparse_pcl,'LOGG3D',out_dim = 256)
    print("[PASSED] LOGG3D")

if __name__ == "__main__":
    voxel = 0.05
    
    from networks.pipelines.pipeline_utils import *
    file = 'tutorial_data/000000.bin'
    run_LOGG3D_test(file,voxel)

    
