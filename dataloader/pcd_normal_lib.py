import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import tqdm

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def viz_distrubtion_and_pcd(point_cloud,data,num_bins,show_normals = False, plot_histogram = True,show_color_bar = False):
    # descretize the polar angles
    bin_edges = np.linspace(data.min(), data.max(), num_bins + 1)  # Divide [0, π] into equal intervals
    bin_indices = np.digitize(data, bin_edges) - 1  # Subtract 1 to make bin indices 0-based
    bin_counts, _ = np.histogram(bin_indices, bins=range(num_bins + 1))

    # Define a colormap for the polar angles (e.g., using the "hsv" colormap)
    colormap = plt.cm.hsv
    colors = colormap(bin_indices / num_bins)[:, :3]  # Normalize bin indices to [0, 1] for colormap

    if plot_histogram:
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(range(num_bins), bin_counts, width=0.8, align='center', alpha=0.7)
        plt.xlabel('Bin Index')
        plt.ylabel('Point Count')
        plt.title('Distribution of Points in Polar Angle Bins')
        plt.xticks(range(num_bins), range(num_bins))
        plt.show()

    if show_color_bar:
        plt.figure(figsize=(10, 6))
        # Create a Normalize object to map values to colors
        norm = Normalize(vmin=data.min(), vmax= data.max())
        sm = ScalarMappable(norm=norm, cmap=colormap)
        sm.set_array([])
        # Create a separate Matplotlib colorbar
        cbar = plt.colorbar(sm, label='Color Scale')
        plt.show()
        
    viz_points(point_cloud,colors,show_normals,normal_scalar=0.1)


def viz_points(point_cloud,colors=None,show_normals = False, normal_scalar = 0.05):
    """
    Visualize a point cloud
    :param point_cloud: An Open3D point cloud object
    :param colors: A NumPy array of shape (N, 3) containing the colors for the point cloud
    :param show_normals: A boolean flag indicating whether to visualize the normals
    :param normal_scalar: A scalar for rescaling the normals for visualization
    :return: None
    """
    # Set the normals
    normals = np.asarray(point_cloud.normals)
    # Rescale the normals for visualization
    normals = normal_scalar* (normals / np.linalg.norm(normals, axis=1, keepdims=True))
    # Update normals
    point_cloud.normals = o3d.utility.Vector3dVector(normals)
    # Assign colors to the point cloud
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Use only RGB components
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud],point_show_normal=show_normals)



def viz_distribution_of_curvature(point_cloud,num_bins,show_normals = False,plot_histogram = True,show_color_bar = False):
    """
    Visualize the distribution of curvature
    :param point_cloud: An Open3D point cloud object
    :return: None
    """
    # Compute the entropy of the normal vectors
    data = compute_curvature(point_cloud)
    viz_distrubtion_and_pcd(point_cloud,data,num_bins,show_normals,plot_histogram,show_color_bar)


def viz_distribution_of_polar(point_cloud,num_bins,show_normals = False,plot_histogram = True,show_color_bar = False):
    '''
    Visualize the point cloud colored based the polar angle
    :param point_cloud: An Open3D point cloud object
    :param polar_bin_indices: A NumPy array of shape (N,) containing the bin indices for the polar angles
    :param num_bins: The number of bins used to discretize the polar angles
    :param show_normals: A boolean flag indicating whether to visualize the normals
    :param plot_histogram: A boolean flag indicating whether to plot the histogram of bin counts
    :param show_color_bar: A boolean flag indicating whether to show the color bar
    :return: None
    '''
    # Compute the entropy of the normal vectors
    _,data,_ = compute_spherical_coord(point_cloud)
    viz_distrubtion_and_pcd(point_cloud,data,num_bins,show_normals,plot_histogram,show_color_bar)
    

def viz_distribution_of_entropy(point_cloud,num_bins,entropy_nn = 20,plot_histogram = True, show_normals = False, show_color_bar = False):
    """
    Visualize the distribution of entropy
    :param point_cloud: An Open3D point cloud object
    :param num_bins: The number of bins used to discretize the entropy
    :param plot_histogram: A boolean flag indicating whether to plot the histogram
    :param show_normals: A boolean flag indicating whether to visualize the normals
    :param show_color_bar: A boolean flag indicating whether to show the color bar
    :return: None
    """
    # Compute the entropy of the normal vectors
    data = compute_normal_entropy(point_cloud,entropy_nn)
    viz_distrubtion_and_pcd(point_cloud,data,num_bins,show_normals,plot_histogram,show_color_bar)


def compute_spherical_coord(point_clouds):
    '''
    Compute the spherical coordinates (r, θ, φ) for each normal vector
    :param point_clouds: An Open3D point cloud object
    :return: A tuple (r, θ, φ) containing the spherical coordinates in degrees
    '''
    normals = np.asarray(point_clouds.normals)
    # Compute the distance from the origin for each normal vector
    r = np.sqrt(normals[:,0]**2 + normals[:,1]**2 + normals[:,2]**2)  # Calculate the distance from the origin
    # Compute the polar angle (θ) for each normal vector
    polar_angles = np.arccos(normals[:,2] / r)  # acos(z / r) gives the angle in radians
    polar_angles_degrees = np.degrees(polar_angles)

    # Compute the azimuthal angle (φ) for each normal vector
    azimuthal_angles = np.arctan2(normals[:, 1], normals[:, 0])
    azimuthal_angles_degrees = np.degrees(azimuthal_angles)
    return  r, polar_angles_degrees, azimuthal_angles_degrees


def compute_normal_entropy_from_open3d(in_pcd,score=0.01,entropy_nn = 20):

    if isinstance(in_pcd,np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(in_pcd)
    else:
        point_cloud = in_pcd
        
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=20))
    #normals = np.asarray(point_cloud.normals)
    data = compute_normal_entropy(point_cloud,entropy_nn)
    indices= np.where(data<score)[0]
    point_cloud = point_cloud.select_by_index(indices)
    return point_cloud

def compute_normal_entropy(point_clouds,entropy_nn = 20,radius=2, max_nn=20):
    '''
    Compute the entropy of the normal vectors
    :param point_clouds: An Open3D point cloud object
    :return: A NumPy array of shape (N,) containing the entropy of the normal vectors
    '''
    if isinstance(point_clouds,np.ndarray):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_clouds)
    else:
        point_cloud = point_clouds
    
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

    # Compute the entropy of the dot products (angular entropy)
    normals = np.asarray(point_cloud.normals)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    points = np.asarray(point_cloud.points)

    entropy = np.zeros(len(normals))
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)
    for i in range(len(points)):
        #n,neighbor_indices,d = kdtree.search_radius_vector_3d(points[i], 0.5)
        n, neighbor_indices, _ = kdtree.search_knn_vector_3d(points[i], entropy_nn)
        if n <2:
            continue
        neighbor_normals = normals[neighbor_indices]
        dot_normal = np.dot(normals[i], neighbor_normals.T)
       
        dot_normal = (dot_normal + 1) / 2 
        #en = -np.sum(dot_normal * np.log10(dot_normal),axis=1)/np.log(10)
        entropy[i] = -np.sum(dot_normal * np.log10(dot_normal))/np.log(10)
    return entropy


def discretized_entropy(pcd,num_bins,entropy_nn = 20,radius=2, max_nn=20,**argv):
    """
    Compute the entropy of the normal vectors
    :param point_clouds: An Open3D point cloud object
    :return: A NumPy array of shape (N,) containing the entropy of the normal vectors
    """
    entropy = compute_normal_entropy(pcd,entropy_nn,radius,max_nn)
    bin_edges = np.linspace(entropy.min(), entropy.max(), num_bins + 1)  # Divide [0, π] into equal intervals
    bin_indices = np.digitize(entropy, bin_edges) - 1  # Subtract 1 to make bin indices 0-based
    #bin_counts, _ = np.histogram(bin_indices, bins=range(num_bins + 1))
    
    discretized = bin_indices / num_bins
    
    return bin_indices

def compute_curvature(point_clouds):
    """
    Compute the curvature of the point cloud
    :param point_clouds: An Open3D point cloud object
    :return: A NumPy array of shape (N,) containing the curvature of the point cloud
    """
    # Normalize the normals for visualization
    points = np.asarray(point_clouds.points)
    normals = np.asarray(point_clouds.normals)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    curvatures = np.zeros(len(points))
    kdtree = o3d.geometry.KDTreeFlann(point_clouds)
    for i in tqdm.tqdm(range(len(points))):
        #n,neighbor_indices,d = kdtree.search_radius_vector_3d(points[i], 2)
        n, neighbor_indices, _ = kdtree.search_knn_vector_3d(points[i], 20)
        if n <2:
            continue
        #_, neighbor_indices, _ = kdtree.search_knn_vector_3d(points[i], 20)
        neighbor_normals = normals[neighbor_indices]
        # Compute the local covariance matrix using the surface normals
        covariance_matrix = np.cov(neighbor_normals.T)
        # Compute the eigenvalues (principal curvatures) of the covariance matrix
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        # Curvature is the average of the magnitudes of the principal curvatures
        curvatures[i] = abs(np.mean(eigenvalues))
    return curvatures


def map_to_positive_z_angle(point_clound):
    normals = np.asarray(point_clound.normals)
    normal_corr = normals.copy()
    bool_select = normals[:,2]<0
    normal_corr[bool_select,:] = normals[bool_select,:]*-1
    # Update the normals
    point_clound.normals = o3d.utility.Vector3dVector(normal_corr)
    return point_clound

def compute_normal_entropy_from_open3d(point_cloud,score=0.01):
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=20))
    normals = np.asarray(point_cloud.normals)
    data = compute_normal_entropy(normals)
    indices= np.where(data<score)[0]
    point_cloud = point_cloud.select_by_index(indices)
    pcd = np.asarray(point_cloud.points)
    return pcd


def extract_entropy(point_clouds):
    """
    Extracts a region of interest (ROI) from a point cloud.

    """
    bag = []
    for pcd in tqdm.tqdm(point_clouds,"Computing Entropy"):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pcd)
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=20))
        
        data = compute_normal_entropy(point_cloud)
        indices= np.where(data<0.01)[0]
        point_cloud = point_cloud.select_by_index(indices)
        pcd = np.asarray(point_cloud.points)
        bag.append(pcd)

    return np.asarray(bag)

# ===============================================================================================


if __name__=="__main__":
    # Loading PCDs to Memory
    #root="data/on-foot"
    root="data/greenhouse"
    root = "log/poses/GreenHouse_e3/0.2_crop_0_entropy_0/map.pcd"
    # Load the parser
    #pcd_loader = load_from_files(root)
    #pcds = pcd_loader.load_pcds(root)

    # Create a new point cloud containing only the horizontal plane points
    #point_cloud = o3d.geometry.PointCloud()
    #point_cloud.points = o3d.utility.Vector3dVector(pcds[45])


    file = "/home/tiago/Dropbox/SHARE/DATASET/r/e3/extracted/map.pcd"
    point_cloud = o3d.io.read_point_cloud(file)

    # search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=10000)
    #point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=20))
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=50))
    point_cloud.normalize_normals()

    point_cloud = map_to_positive_z_angle(point_cloud)
    # ===============================================================================================
    # Visualize the point cloud
    #viz_points(point_cloud,show_normals=True)
    
    # Visualize the point cloud colored based the polar angle
    #viz_distribution_of_polar(point_cloud,20,plot_histogram = False,show_normals= True)
    
    # Visualize the distribution of entropy
    #viz_distribution_of_entropy(point_cloud,20,plot_histogram = False,show_normals = True)
    
    #normals = np.asarray(point_cloud.normals)
    #pcds_np = np.array(point_cloud.points)
    #point_clouds_np = extract_roi(pcds_np,[-20,-20,-20],[20,20,20])
    #point_cloud = utils.velo_utils.np_pts_to_pcd(pcds_np)

    #viz_points(point_cloud, show_normals=False)
    
    #exit()
    
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=20))
    #data = compute_normal_entropy(point_cloud,30)
    
    #for i in range(1,50,5):
    #    entropy_thresh_score = float(i/50)
    #    print(entropy_thresh_score)
    #    indices= np.where(data<entropy_thresh_score)[0]
    #    sub_set_point_cloud = point_cloud.select_by_index(indices)
        
        #pcds_np = np.array(sub_set_point_cloud.points)
        #point_clouds_np = extract_roi(pcds_np,[-20,-20,-20],[20,20,20])
    #    point_cloud_down_sampled = o3d.geometry.PointCloud()
    #    point_cloud_down_sampled.points = o3d.utility.Vector3dVector(np.array(sub_set_point_cloud.points))

        #viz_points(sub_set_point_cloud, show_normals=False)

    viz_distribution_of_entropy(point_cloud,10,plot_histogram = False,show_normals = False)

    #viz_distribution_of_curvature(point_cloud,20,plot_histogram = False,show_normals = True)