

import torch
import numpy as np
import open3d as o3d

def compute_normals_open3d(point_cloud, radius=0.1,max_nn=10):
    """
    Compute normals for a point cloud using Open3D.
    
    Args:
    - point_cloud (numpy.ndarray): Input point cloud, shape (N, 3) where N is the number of points.
    - radius (float): Radius for normal estimation.
    
    Returns:
    - normals (numpy.ndarray): Normals for each point in the point cloud, shape (N, 3).
    """
    
    if not isinstance(point_cloud, np.ndarray):
        raise ValueError("Input point cloud must be a numpy array.")
    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    
    # Retrieve normals
    normals = np.asarray(pcd.normals)
    
    return normals



def compute_normals(point_cloud, k=20,radius=2):
    """
    Compute normals for a point cloud.
    
    Args:
    - point_cloud (torch.Tensor): Input point cloud, shape (N, 3) where N is the number of points.
    - k (int): Number of neighbors to consider for computing normals.
    
    Returns:
    - normals (torch.Tensor): Normals for each point in the point cloud, shape (N, 3).
    """
    # Find k-nearest neighbors for each point
    from torch_geometric.nn import knn
    _, idx = knn(point_cloud, point_cloud, k=k,)
    # Knn based on radius
    from torch_geometric.nn import radius as radius_knn
    #_,idx = radius_knn(point_cloud, point_cloud,radius)
    
    idx = idx.reshape(point_cloud.shape[0],-1)
    print(idx)
    
    
    # Compute covariance matrix for each point
    device = point_cloud.device
    cov_matrices = []
    for i in range(point_cloud.shape[0]):
        neighbors = point_cloud[idx[i]]
        center = point_cloud[i].unsqueeze(0)
        centered_neighbors = neighbors - center
        cov_matrix = torch.matmul(centered_neighbors.t(), centered_neighbors) / k
        cov_matrices.append(cov_matrix)
    cov_matrices = torch.stack(cov_matrices)
    
    # Compute eigenvalues and eigenvectors of covariance matrices
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrices)
    
    
    # Choose the smallest eigenvalue and corresponding eigenvector as normal
    normal_indices = torch.argmin(eigenvalues, dim=1)
    normals = eigenvectors[torch.arange(point_cloud.shape[0]), normal_indices]
    
    return normals



if __name__ == "__main__":
    # Test compute_normals
    point_cloud = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float)
    # generate point cloud from a surface   
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    x, y = torch.meshgrid(x, y)
    z = x**2 + y**2
    point_cloud = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    
    normals = compute_normals(point_cloud)
    print(normals)
    
    # %%
    # plot point cloud and normals
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    ax.quiver(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], normals[:, 0], normals[:, 1], normals[:, 2], color='r', length=0.05)
    plt.show()
    # save figure
    plt.savefig("point_cloud_normals.png")
    # Expected output: tensor([[ 0.0000,  0.0000,  1.0000],
    #                           [ 0.0000,  0.0000,  1.0000],
    #                           [ 0.0000,  0.0000,  1.0000],
    #                           [ 0.0000,  0.0000,  1.0000]])
    # %%
    
    # %%