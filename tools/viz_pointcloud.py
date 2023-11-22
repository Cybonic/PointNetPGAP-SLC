#!/usr/bin/env python3

import yaml
import os
from tqdm import tqdm
import argparse
from dataloader.ORCHARDS import OrchardDataset
from dataloader.KITTI import KittiDataset
from dataloader.POINTNETVLAD import PointNetEval
from PyQt5 import QtWidgets
import open3d as o3d
import time
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def load_dataset(dataset):
    if dataset == 'KITTI':
        dataset = KittiDataset
    elif dataset.lower() == 'orchard-uk':
       dataset = OrchardDataset
    elif dataset.lower() == 'pointnetvlad':
       dataset = PointNetEval
    else:
        raise NameError
    
    return dataset

def scatter_plot(loader):
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  #for i in range(len(loader)):
  pcl = loader(0)
  x = pcl[:,0]
  y = pcl[:,1]
  z = pcl[:,2]
  x = x[z>0]
  y = y[z>0]
  z = z[z>0]
  
  ax.scatter(x,y,z)
  # ax.set_aspect('auto')
  plt.show()
   

class ViewerWidget(QtWidgets.QWidget):
  def __init__(self):

    # https://stackoverflow.com/questions/58732865/pointcloud2-stream-visualization-in-open3d-or-other-possibility-to-visualize-poi
    self.vis = o3d.visualization.Visualizer()
    self.point_cloud = None
    self.vis.create_window()

  ############################################################################
  def updater(self,loader):
    pcd = o3d.geometry.PointCloud()
    for i in range(len(loader)):
      pcl = loader(i)
      z = pcl[:,2]
      pcl = pcl[z>0,:]
 
      pcd.points = o3d.utility.Vector3dVector(pcl)
      if i == 0:
        print('get points')
        self.vis.add_geometry(pcd, reset_bounding_box=True)
        print ('add points')

      self.vis.update_geometry(pcd)
      self.vis.poll_events()
      self.vis.update_renderer()
      time.sleep(0.1)


if __name__ == '__main__':

  parser = argparse.ArgumentParser("./infer.py")
  parser = argparse.ArgumentParser(description='Play back images from a given directory')
  parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
  parser.add_argument('--dynamic',default  = 1 ,type = int)
  parser.add_argument('--dataset',
                                  default = 'pointnetvlad',
                                  type=str,
                                  help=' dataset root directory .'
                                  )
  parser.add_argument('--seq',    
                              default  = ['00'],
                              type = str)
  parser.add_argument('--plot',default  = True ,type = bool)
  parser.add_argument('--loop_thresh',default  = 1 ,type = float)
  parser.add_argument('--record_gif',default  = False ,type = bool)
  parser.add_argument('--cfg',default  = 'cfg/overlap_cfg.yaml' ,type = str)
  parser.add_argument('--option',default  = 'compt' ,type = str,choices=['viz','compt'])
  parser.add_argument('--pose_file',default  = 'poses' ,type = str)
  
  args = parser.parse_args()

  root    = args.root
  dataset = args.dataset 
  seq     = args.seq
  plotting_flag = args.plot
  record_gif_flag = args.record_gif
  cfg_file = args.cfg
  option = args.option
  loop_thresh = args.loop_thresh

  FLAGS, unparsed = parser.parse_known_args()

 
  

  ground_truth = {'pos_range':10, # Loop Threshold [m]
                  'neg_range': 17,
                  'num_neg':20,
                  'num_pos':50,
                  'warmupitrs': 600, # Number of frames to ignore at the beguinning
                  'roi':500}
  
  dataset    = FLAGS.dataset
  sequence   = FLAGS.seq
  #max_points = FLAGS.max_points

  dataset_loader = load_dataset(dataset)

  session_cfg_file = os.path.join('sessions', args.dataset.lower() + '.yaml')
  print("Opening session config file: %s" % session_cfg_file)
  SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
  dataset = SESSION['val_loader']['data']['dataset']
  sequence = SESSION['val_loader']['data']['sequence']
  loader = dataset_loader(SESSION['root'],dataset,sequence,ground_truth=ground_truth,modality = 'pcl',square_roi = [{'xmin':-15,'xmax':15,'ymin':-15,'ymax':15,'zmax':1}])

  print("----------")
  print("INTERFACE:")
  print("Root: ", SESSION['root'])
  print("Dataset  : ", FLAGS.dataset)
  print("Sequence : ",FLAGS.seq)
  # print("Max points : ",FLAGS.max_points)
  print("----------\n")

  pcl = ViewerWidget()
  pcl.updater(loader)
  
  