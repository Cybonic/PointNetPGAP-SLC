
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.laserscan import Scan
from PIL import Image
import numpy as np
import torch

def test_laserscan_tensor(file,max_points=10000):
    data_handler = Scan(max_points=10000) # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    points,intensity = data_handler.load(file)
    input = data_handler.to_tensor(points)
    assert torch.is_tensor(input), "data should be from the SparseTensor type"
    n,c = input.shape
    assert max_points==n
    return True,input

def run_laserscan_test(file):
    flag,sparsetensor = test_laserscan_tensor(file)
    print("[PASSED] LaserScan")
    
if __name__ == "__main__":
    voxel_size=0.05
    file = 'tutorial_data/000000.bin'
    run_laserscan_test(file)

