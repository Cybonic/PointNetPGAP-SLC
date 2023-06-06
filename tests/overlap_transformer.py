
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from spherical_projection import test_spherical
from dataloader.projections import SphericalProjection
from networks.pipeline_factory import get_pipeline
from PIL import Image
import numpy as np
import torch

def test_transformer(input:np.ndarray,model_name:str,out_dim:int):
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
    
    assert torch.is_tensor(input),"Input should be in tensor format"
    batch = torch.stack((input,input), dim=0)

    model = get_pipeline(model_name,output_dim=out_dim)
    assert str(model) == model_name,'Name does not match'
    
    descriptor = model(batch)
    assert torch.is_tensor(descriptor), "output should be a tensor"
    b,d = descriptor.shape
    assert b == 2,'Batch size is wrong'
    assert d == out_dim, "descriptor shape is wrong"

    return True, descriptor

width = 900
height = 64

file = 'tutorial_data/000000.bin'
flag,proj_height = test_spherical(file,width,height)
print("Spherical is passing")

flag,dptrs = test_transformer(proj_height,'overlap_transformer',out_dim = 255)
print("overlap_transformer is working correctly")
