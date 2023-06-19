import numpy as np
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate
from torchsparse import SparseTensor
from .laserscan import LaserScan
import torch


def numpy_to_sparse(points,voxel_size,n_points=None):
    # get rounded coordinates
    coords = np.round(points[:,:3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = points

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    # Impose a fixe size
    indices = indices if n_points == None else indices[:n_points]
       
    #print(indices.shape[0])
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    inputs = sparse_collate([inputs])
    inputs.C = inputs.C.int()

    return inputs


class SparseLaserScan(LaserScan):
    def __init__(self,voxel_size, parser = None, max_points = -1, aug_flag=False):
        super(SparseLaserScan,self).__init__(parser,max_points,aug_flag)
        self.voxel_size = voxel_size
    
    def load(self,file):
        self.open_scan(file)
        filtered_points,filtered_remissions = self.get_points()
        return filtered_points,filtered_remissions

    def to_sparse_tensor(self,points):
        return numpy_to_sparse(points,self.voxel_size)
    
    def __call__(self,files):
        buff = []
        if not isinstance(files,list) and not isinstance(files,np.ndarray):
            files = [files]
        
        for file in files: 
            points,intensity = self.load(file)
            pcl = np.concatenate((points,intensity.reshape(-1,1)),axis=-1)
            buff.append(pcl)

        return self.to_sparse_tensor(pcl)
    
class Scan(LaserScan):
  def __init__(self,parser = None, max_points = -1, aug_flag=False):
    super(Scan,self).__init__(parser,max_points,aug_flag)
    pass

  def load(self,file):
    self.open_scan(file)
    filtered_points,filtered_remissions = self.get_points()
    return filtered_points,filtered_remissions
  
  def to_tensor(self,input):
    input = torch.tensor(input).type(torch.float32)
    #input = input.transpose(dim0=1,dim1=0)
    return input
  
  def __call__(self,files):

    points,intensity = self.load(files)

    return self.to_tensor(points)

  def __str__(self):
    return "SparseTensor"

  