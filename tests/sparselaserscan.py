
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.sparselaserscan import SparseLaserScan
from PIL import Image
import numpy as np
from torchsparse import SparseTensor

def test_sparse_tensor(file,voxel_size):
    data_handler = SparseLaserScan(voxel_size=voxel_size) # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    points,intensity = data_handler.load(file)
    pcl = np.concatenate((points,intensity.reshape(-1,1)),axis=-1)
    input = data_handler.to_sparse_tensor(pcl)
    assert isinstance(input,SparseTensor), "data should be from the SparseTensor type"
    return True,input

def run_sparse_test(file,voxel_size):
    flag,sparsetensor = test_sparse_tensor(file,voxel_size)
    print("[PASSED] SparseTensor")



if __name__ == "__main__":
    voxel_size=0.05
    file = 'tutorial_data/000000.bin'
    run_sparse_test(file,voxel_size)

