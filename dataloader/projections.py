#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
from .laserscan import LaserScan
import torchvision.transforms as Tr
import torch 

PREPROCESSING = Tr.Compose([Tr.ToTensor()])

class SphericalProjection(LaserScan):
  def __init__(self, width = 900,height=64, fov_up=3.0, fov_down=-25.0, 
                max_range=50,max_rem=1,
                parser = None, max_points = -1, aug_flag=False):
    '''
    width,
    height,
    parser = None, 
    max_points = -1, -> all points
    aug_flag=False
    '''

    super(SphericalProjection,self).__init__(parser,max_points,aug_flag)
    self.width = width
    self.height = height
    self.fov_up = fov_up
    self.fov_down = fov_down
    self.max_rem = max_rem
    self.max_depth = max_range

  def to_tensor(self,input):
    return PREPROCESSING(input).type(torch.float32)
  
  def load(self,file):
    self.open_scan(file)
    filtered_points,filtered_remissions = self.get_points()
    return range_projection(filtered_points, filtered_remissions,
                            fov_up=self.fov_up, 
                            fov_down=self.fov_down, 
                            proj_H=self.height, 
                            proj_W= self.width, 
                            max_range=self.max_depth)

  def __call__(self,files):

    image = self.load(files)
    values = list(image.values())
    image = np.concatenate(values,axis=-1)
    return self.to_tensor(image)
  
  def __str__(self):
    return "spherical"


class BEVProjection(LaserScan):
  def __init__(self,width,height, parser = None, max_points = -1, aug_flag=False,**args):
    '''
    width,
    height,
    parser = None, 
    max_points = -1, -> all points
    aug_flag=False
    '''

    super(BEVProjection,self).__init__(parser,max_points,aug_flag,**args)
    self.width = width
    self.height = height

  def to_tensor(self,input):
    return PREPROCESSING(input).type(torch.float32)
  
  def load(self,file):
    self.open_scan(file)
    return self.get_points()
     
  
  def __call__(self,file,set_augmentation=False,**argv):
    points,remissions  = self.load(file)
    
    if set_augmentation:
        points = self.set_augmentation(points)
        
    image = get_bev_proj(points,remissions,self.width,self.height)
    
    values = list(image.values())
    image = np.concatenate(values,axis=-1)
    return self.to_tensor(image)
  
  def __str__(self):
    return "bev"


def get_bev_proj(points,remissions,Width,Height):
  """ Project a pointcloud into a spherical projection image.projection.
      Function takes no arguments because it can be also called externally
      if the value of the constructor was not set (in case you change your
      mind about wanting the projection)
  """

  to_image_space = True # parameters['to_image_space'] # Map the depth values to [0,255]
  # Get max and min values along the zz axis
  zmax = points[:,2].max()
  zmin = points[:,2].min()

  xmax = points[:,0].max()
  xmin = points[:,0].min()

  ymax = points[:,1].max()
  ymin = points[:,1].min()


  discretization_x = (xmax - xmin)/(Height)
  discretization_y = (ymax - ymin)/(Width)
  

  # Discretize Feature Map
  PointCloud = np.nan_to_num(np.copy(points))
  Intensity = np.nan_to_num(np.copy(remissions))


  PointCloud[:, 0] = np.int_(np.floor(PointCloud[:, 0] / discretization_x)  + (Height)/2).clip(min=0,max=(Height-1))
  PointCloud[:, 1] = np.int_(np.floor(PointCloud[:, 1] / discretization_y) + (Width)/2).clip(min=0,max=(Width-1))

  # sort-3times
  indices = np.lexsort((-PointCloud[:, 2], PointCloud[:, 1], PointCloud[:, 0]))
  PointCloud = PointCloud[indices]

  # Height Map
  heightMap = np.zeros((Height, Width))

  _, indices = np.unique(PointCloud[:, 0:2], axis=0, return_index=True)
  PointCloud_frac = PointCloud[indices]

  xx_idx = np.int_(PointCloud_frac[:, 0])
  yy_idx = np.int_(PointCloud_frac[:, 1])

  #heightMap[xx_idx,yy_idx] = PointCloud_frac[:, 2]
  heightMap = PointCloud_frac[:, 2]

  # Intensity Map & DensityMap
  intensityMap = np.zeros((Height, Width))
  densityMap = np.zeros((Height, Width))

  _, indices, counts = np.unique(PointCloud[:, 0:2], axis=0, return_index=True, return_counts=True)
  
  PointCloud_top = PointCloud[indices]
  Intensity_top = Intensity[indices]

  intensityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = Intensity_top

  normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
  densityMap[np.int_(PointCloud_top[:, 0]), np.int_(PointCloud_top[:, 1])] = normalizedCounts

  if to_image_space == True:
    # Density
    densityMap = densityMap*255
    densityMap = np.clip(densityMap,a_min=0,a_max=255).astype(np.uint8)
    # Intensity
    intensityMap = (np.clip(intensityMap,a_min = 0, a_max = 100)/100) *255
    intensityMap = np.clip(intensityMap,a_min=0,a_max=255).astype(np.uint8)
    # Height

    heightMap = np.clip(heightMap,a_max = zmax,a_min = zmin)
    heightMap = (((heightMap-zmin)/float(zmax-zmin))*255).astype(np.uint8)
    heightMap_out = np.zeros((Height, Width))
    heightMap_out[xx_idx,yy_idx] = heightMap
    #heightMap = np.clip(heightMap,a_min=0,a_max=255).astype(np.uint8)
  
  density = np.expand_dims(densityMap,axis=-1)
  height  = np.expand_dims(heightMap_out,axis=-1)
  intensity = np.expand_dims(intensityMap,axis=-1)

  return {'height': height, 'density': density, 'intensity': intensity}
    
  

def get_spherical_proj(points,remissions,fov_up,fov_down,Width,Height,max_rem,max_depth):
  """ Project a pointcloud into a spherical projection image.projection.
      Function takes no arguments because it can be also called externally
      if the value of the constructor was not set (in case you change your
      mind about wanting the projection)
  """
  # Get the required parameters
  proj_fov_up = fov_up
  proj_fov_down = fov_down
  proj_W = Width
  proj_H = Height
  max_rem = max_rem
  max_depth = max_depth
  to_image_space = True # parameters['to_image_space'] # Map the depth values to [0,255]

  # points,remissions = self.get_points()

  # projected range image - [H,W] range (-1 is no data)
  proj_range = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)

  # unprojected range (list of depths for each point)
  unproj_range = np.zeros((0, 1), dtype=np.float32)

  # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
  proj_xyz = np.full((proj_H, proj_W, 3), -1,
                          dtype=np.float32)

  # projected remission - [H,W] intensity (-1 is no data)
  proj_remission = np.full((proj_H, proj_W), -1,
                                dtype=np.float32)

  # projected index (for each pixel, what I am in the pointcloud)
  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                          dtype=np.int32)

  # for each point, where it is in the range image
  proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
  proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

  # mask containing for each pixel, if it contains a point or not
  proj_mask = np.zeros((proj_H, proj_W),
                            dtype=np.int32)       # [H,W] mask


  # laser parameters
  fov_up = proj_fov_up / 180.0 * np.pi      # field of view up in rad
  fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

  # get depth of all points
  depth = np.linalg.norm(points, 2, axis=1)
  depth = np.nan_to_num(depth)

  # get scan components
  scan_x = np.nan_to_num(points[:, 0])
  scan_y = np.nan_to_num(points[:, 1])
  scan_z = np.nan_to_num(points[:, 2])

  # get angles of all points
  #yaw = -np.arctan2(scan_y, scan_x)
  yaw = np.nan_to_num(np.arctan2(scan_y, scan_x))
  pitch = np.arcsin(scan_z / depth)
  pitch = np.nan_to_num(pitch)
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

  # scale to image size using angular resolution
  proj_x *= proj_W                              # in [0.0, W]
  proj_y *= proj_H                              # in [0.0, H]

  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
  proj_x = np.copy(proj_x)  # store a copy in orig order

  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
  proj_y = np.copy(proj_y)  # stope a copy in original order

  # copy of depth in original order
  unproj_range = np.copy(depth)

  # order in decreasing depth
  indices = np.arange(depth.shape[0])
  order = np.argsort(depth)[::-1]

  depth = depth[order]
  indices = indices[order]

  points = points[order]
  remission = remissions[order]
  
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  # assing to images
  proj_range[proj_y, proj_x] = depth
  proj_xyz[proj_y, proj_x] = points
  proj_remission[proj_y, proj_x] = remission
  proj_idx[proj_y, proj_x] = indices
  proj_mask = (proj_idx > 0).astype(np.int32)

  if to_image_space == True:
    proj_range =(np.clip(proj_range,a_min=0,a_max=max_depth)/max_depth)*255
    proj_range = proj_range.clip(max=255).astype(np.uint8)
    
    proj_remission = (np.clip(proj_remission,a_min=0,a_max=max_rem)/max_rem)*255
    proj_remission = proj_remission.clip(max=255).astype(np.uint8)

  # Expand dim to create a matrix [H,W,C]
  proj_range = np.expand_dims(proj_range,axis=-1)
  proj_remission = np.expand_dims(proj_remission,axis=-1)
  proj_idx = np.expand_dims(proj_idx,axis=-1)
  proj_mask = np.expand_dims(proj_mask,axis=-1)
  
  return {'range': proj_range, 'xyz': proj_xyz, 'remission': proj_remission,'idx':proj_idx,'mask': proj_mask}



def range_projection(current_vertex, intensity, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = intensity
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity

  # Expand dim to create a matrix [H,W,C]
  proj_range = np.expand_dims(proj_range,axis=-1)
  proj_intensity = np.expand_dims(proj_intensity,axis=-1)
  proj_idx = np.expand_dims(proj_idx,axis=-1)
  
  return {'range': proj_range, 'xyz': proj_vertex, 'remission': proj_intensity,'idx':proj_idx}
  # return proj_range, proj_vertex, proj_intensity, proj_idx




  

    
