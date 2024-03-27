#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import numpy as np
from scipy.spatial.transform import Rotation as R
import random,math
import torchvision.transforms as Tr
import torch
PREPROCESSING = Tr.Compose([Tr.ToTensor()])

def shuffle_points(points):
  """
  Shuffle the points
  """
  indices = np.arange(len(points))
  np.random.shuffle(indices)
  return points[indices]

def random_subsampling(points,max_points):
  '''
  https://towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c

  '''
  if not isinstance(points, np.ndarray):
    points = np.array(points)

  n_points = points.shape[0]
  #print(n_points)
  #print(max_points)
  assert n_points>=max_points,'Max Points is to big'
  sample_idx = np.random.randint(0,n_points,max_points)
  sample_idx  =np.sort(sample_idx)
  return(sample_idx)



def square_roi(PointCloud,roi_array):
  mask_list = []
  for roi in roi_array:
    region_mask_list = []
    assert isinstance(roi,dict),'roi should be a dictionary'

    if 'xmin' in roi:
      region_mask_list.append((PointCloud[:, 0] >= roi["xmin"]))
    if 'xmax' in roi:
      region_mask_list.append((PointCloud[:, 0]  <= roi["xmax"]))
    if 'ymin' in roi:
      region_mask_list.append((PointCloud[:, 1] >= roi["ymin"]))
    if 'ymax' in roi:
      region_mask_list.append((PointCloud[:, 1]  <= roi["ymax"]))
    if 'zmin' in roi:
      region_mask_list.append((PointCloud[:, 2] >= roi["zmin"]))
    if 'zmax' in roi:
      region_mask_list.append((PointCloud[:, 2]  <= roi["zmax"]))
    
    mask = np.stack(region_mask_list,axis=-1)
    mask = np.product(mask,axis=-1).astype(np.uint8)
    mask_list.append(mask)

    
  mask = np.stack(mask_list,axis=-1)
  mask = np.sum(mask,axis=-1).astype(np.uint8)
  return(mask)


def cylinder_roi(PointCloud,roi_array):

  mask_list = []
  for roi in roi_array:
    region_mask_list = []
    assert isinstance(roi,dict),'roi should be a dictionary'
    dist = np.linalg.norm(PointCloud[:, 0:3],axis=1) 
    if 'rmin' in roi:
      region_mask_list.append((dist >= roi["rmin"]))
    if 'rmax' in roi:
      region_mask_list.append((dist < roi["rmax"]))
    
    mask = np.stack(region_mask_list,axis=-1)
    mask = np.product(mask,axis=-1).astype(np.bool)
    mask_list.append(mask)
  
  mask = np.stack(mask_list,axis=-1)
  mask = np.sum(mask,axis=-1).astype(np.bool)
  return(mask)


def normalize_pcl(points):
  """
  Normalize the point cloud to fit in a unit sphere
  https://medium.com/@kidargueta/normalizing-feature-scaling-point-clouds-for-machine-learning-8138c6e69f5
  
  """
  
  centroid = np.mean(points, axis=0)
  points -= centroid
  furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
  points /= furthest_distance
  return points

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, parser = None, max_points = -1, aug=False,clean_zeros = False, **argv):

    self.reset()
    self.parser = parser
    self.clean_zeros = clean_zeros

    self.set_pcl_norm_flag = False
    if 'pcl_norm' in argv and argv['pcl_norm'] == True:
      self.set_pcl_norm_flag = True
      
    self.max_points = max_points
    self.noise      = 0
    self.set_aug_flag = aug
    
    # Configure ROI
    self.roi = {}
    self.set_roi_flag = False
    if 'square_roi' in argv and argv['square_roi'] != None:
      self.set_roi_flag = True
      self.roi['square_roi'] = argv['square_roi']
    
    if 'cylinder_roi' in argv and argv['cylinder_roi'] != None:
      self.set_roi_flag = True
      self.roi['cylinder_roi'] = argv['cylinder_roi']


  def reset(self):
    """ Reset scan members. """
    #self.roi  = None 
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

   
  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]


  def __len__(self):
    return self.size()


  def open_scan(self, filename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    if self.parser != None:
      scan = self.parser.velo_read(filename)
    else: 
      scan = np.fromfile(filename, dtype=np.float32)
      scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission

    self.set_points(points, remissions)


  def load_pcl(self,scan):
    # Read point cloud already loaded
    self.reset()
    points = scan[:, 0:3]    # get xyz
    remissions = np.zeros(scan.shape[0])
    if scan.shape[1]==4:
      remissions = scan[:, 3]  # get remission
    self.set_points(points, remissions)


  def set_points(self, points, remissions):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)


  # ==================================================================
  def set_roi(self):
    PointCloud = self.points.copy()
    Remissions = self.remissions
    n_points= PointCloud.shape[0]

    mask = np.ones(n_points,dtype=np.uint8) # By default use all the points
    # Each roi extraction approach considers the entire point cloud 
    if 'square_roi' in self.roi:
      roi = self.roi['square_roi']
      if isinstance(roi,dict):
        roi = [roi]

      local_mask = square_roi(PointCloud,roi)
      mask = np.logical_and(mask,local_mask)
    
    if 'cylinder_roi' in self.roi:
      roi = self.roi['cylinder_roi']
      assert isinstance(roi,list)
      local_mask = cylinder_roi(PointCloud,roi)
      mask = np.logical_and(mask,local_mask)

    self.points = PointCloud[mask,:]
    self.remissions = Remissions[mask]

  # ==================================================================
  # ==================================================================
  def set_augmentation(self,points):
    # https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263

    pointcloud = points.copy()
    # choose a random angle and rotate the point cloud along the z-axis from -angle_range to angle_range
    
    angle_range = [0, math.pi,math.pi/2,-math.pi/2, math.pi]
    # select an angle from the angle_range array uniformly
    theta = random.choice(angle_range)
    
    #theta = random.random() * 2. * math.pi # rotation angle
    rot_matrix = np.array([[math.cos(theta), -math.sin(theta),    0],
                          [ math.sin(theta),  math.cos(theta),    0],
                          [0,                             0,      1]])

    points = rot_matrix.dot(pointcloud.T).T
    
    return points

   
  def set_noise(self):
    # add some noise
    noise = np.random.normal(0,self.noise, (self.points.shape))
    noisy_pointcloud = self.points + noise
    self.points = noisy_pointcloud
    
  # ==================================================================
  def set_sampling(self):

    idx = random_subsampling(self.points,self.max_points)
    self.points = self.points[idx,:]
    self.remissions = self.remissions[idx]


  # ==================================================================
  def get_points(self):
    
    if self.clean_zeros:
      mask = np.logical_and(self.points[:,0]!=0,self.points[:,1]!=0,self.points[:,2]!=0)
      self.points = self.points[mask,:]
      self.remissions = self.remissions[mask]
      
      
    if self.set_roi_flag:
      self.set_roi()

    if self.max_points > 0:  # if max_points == -1 do not subsample
      self.set_sampling()

    #if self.set_aug_flag:
    #  self.set_augmentation()

    if self.set_pcl_norm_flag:
      self.points = normalize_pcl(self.points)
    
    return self.points,self.remissions
  





class Scan(LaserScan):
  def __init__(self,parser = None, max_points = -1,**argv):
    super(Scan,self).__init__(parser,max_points,**argv)
    
    self.shuffle_points = True if 'shuffle_points' in argv and argv['shuffle_points'] == True else False
    pass

  def load(self,file):
    self.open_scan(file)
    filtered_points,filtered_remissions = self.get_points()
    return filtered_points,filtered_remissions
  
  def to_tensor(self,input):
    input = torch.tensor(input).type(torch.float32)
    #input = input.transpose(dim0=1,dim1=0)
    return input
  
  def __getitem__(self,file):
    points, intensity = self.load(file)
    return points
  
  def __call__(self,files,set_augmentation=False,set_shuffle_points=False):
    points,intensity = self.load(files)
    
    if set_augmentation:
      points = self.set_augmentation(points)
    if set_shuffle_points:
      points = shuffle_points(points)
            
    return self.to_tensor(points)
  
  def __str__(self):
    return "pcl"
