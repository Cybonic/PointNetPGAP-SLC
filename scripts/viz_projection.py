#!/usr/bin/env python3

from tqdm import tqdm
import argparse
import numpy as np

from dataloader.KITTI import KittiDataset
import numpy as np
from torchvision import transforms as T
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import imageio
from PIL import Image,ImageOps
import os
import yaml

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

if __name__ == "__main__":
    
  parser = argparse.ArgumentParser(description='Play back images from a given directory')
  parser.add_argument('--root', type=str, default='/home/tiago/Dropbox/research/datasets')
  parser.add_argument('--dynamic',default  = 1 ,type = int)
  parser.add_argument('--dataset',
                                  default = 'pointnetvlad',
                                  type=str,
                                  help=' dataset root directory .'
                                  )
  parser.add_argument('--seq',    
                              default  = 'summer',
                              type = str)
  parser.add_argument('--plot',default  = True ,type = bool)
  parser.add_argument('--loop_thresh',default  = 1 ,type = float)
  parser.add_argument('--record_gif',default  = False ,type = bool)
  parser.add_argument('--option',default  = 'compt' ,type = str,choices=['viz','compt'])
  parser.add_argument('--pose_file',default  = 'poses' ,type = str)
  
  args = parser.parse_args()

  root    = args.root
  dataset = args.dataset 
  seq     = args.seq
  plotting_flag = args.plot
  record_gif_flag = args.record_gif
  option = args.option
  loop_thresh = args.loop_thresh

  print("[INF] Dataset Name:  " + dataset)
  print("[INF] Sequence Name: " + str(seq) )
  print("[INF] Plotting Flag: " + str(plotting_flag))
  print("[INF] record gif Flag: " + str(record_gif_flag))
  print("[INF] Reading poses from : " + args.pose_file)

  ground_truth = {'pos_range':10, # Loop Threshold [m]
                  'neg_range': 17,
                  'num_neg':20,
                  'num_pos':50,
                  'warmupitrs': 600, # Number of frames to ignore at the beguinning
                  'roi':500}
  
  dataset_loader = load_dataset(dataset)
  
  session_cfg_file = os.path.join('sessions', args.dataset.lower() + '.yaml')
  print("Opening session config file: %s" % session_cfg_file)
  SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
  dataset = SESSION['val_loader']['data']['dataset']
  sequence = SESSION['val_loader']['data']['sequence']
  loader = dataset_loader(SESSION['root'],dataset,sequence,ground_truth=ground_truth,modality = 'bev',square_roi = [{'xmin':-15,'xmax':15,'ymin':-15,'ymax':15,'zmax':1}])
 

  #loader = OrchardDataset(root,'',sequence,sync = True,modality = 'bev',square_roi = [{'xmin':-15,'xmax':15,'ymin':-15,'ymax':15,'zmax':1}]) #cylinder_roi=[{'rmax':10}])

  fig = Figure(figsize=(5, 4), dpi=25,)
  fig, ax = plt.subplots()

  filename = 'projection.gif'
  canvas = FigureCanvasAgg(fig)
  writer = imageio.get_writer(filename, mode='I')

  fig, ax = plt.subplots(1, 1)
  num_samples = len(loader)

  for i in tqdm(range(500,num_samples,10)):
    
    input = loader._get_modality_(i)[:,:,0] # Get only Bev projection
    input_im = input # .astype(np.uint8)
  
    pil_range = Image.fromarray(input_im.astype(np.uint8))
    pil_range = ImageOps.colorize(pil_range, black="white", white="black")

    X = np.asarray(pil_range)
    writer.append_data(X)






  
  