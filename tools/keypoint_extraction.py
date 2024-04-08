


import open3d as o3d
import os
import numpy as np
import json
def extract_keypoints(pcd, method='ISS'):
    if method == 'ISS':
        keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd)
    elif method == 'SIFT':
        keypoints = o3d.geometry.keypoint.compute_sift_keypoints(pcd)
    elif method == 'HARRIS':
        keypoints = o3d.geometry.keypoint.compute_harris_keypoints(pcd)
    elif method == 'FAST':
        keypoints = o3d.geometry.keypoint.compute_fast_keypoints(pcd)
    else:
        raise ValueError('Invalid method')
    return keypoints
dataset = os.path.join('/home/tiago/workspace/DATASET/HORTO-3DLM/ON22/extracted/point_cloud/')

# find all pcd files in the directory
files = np.sort([os.path.join(dataset,f) for f in os.listdir(dataset) if f.endswith('.bin')])

print(len(files))


for file in files:
    # load bin file using numpy
    pcd = np.fromfile(file, dtype=np.float32)
    pcd = pcd.reshape(-1, 4)
    
    xyz,r = pcd[:,0:3],pcd[:,3]
    
    # create open3d point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    key_points = extract_keypoints(pcd, method='ISS')
    
    # plot keypoints on the point cloud
    key_points.paint_uniform_color([1, 0, 0])

    # plot point cloud and keypoints  on png 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(key_points)
    vis.run()
    vis.capture_screen_image("point_cloud.png")
    vis.destroy_window()
    
    




