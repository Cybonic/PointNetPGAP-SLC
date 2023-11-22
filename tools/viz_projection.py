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

    modality = "bev"

    # These networks use proxy representation to encode the point clouds
    if modality == "bev":
        modality = BEVProjection(256,256)
    elif modality == "spherical":
        modality = SphericalProjection(256,256)


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






  
  