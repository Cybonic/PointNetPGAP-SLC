

# Show point cloud on torch geometry
#
# This script shows how to visualize a point cloud using torch geometry.

import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_pcl(pcl):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pcl.pos[:, 0], pcl.pos[:, 1], pcl.pos[:, 2])
    plt.show()
    plt.savefig('point_cloud.png')
    
# Create a point cloud
pos = torch.Tensor(np.random.rand(10000, 3))
pcl = Data(pos=pos)

# Show the point cloud
show_pcl(pcl)
