
from genericpath import isdir
import os 
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


from dataloader.kitti import kitti 
import yaml
from utils.loss import L2_np
from utils.viz import plot_retrieval_on_map,color_similarity_on_map
from utils.utils import generate_descriptors
from networks import model
import numpy as np
import argparse
import torch
from utils.viz import myplot
from tqdm import tqdm


def plot_sim_on_map(descriptors,anchor,map_idx,poses,record_gif=False):
    # Save Similarity map
    root = '/media/tiago/BIG/Orchards/sim$'
    if os.path.isdir(root):
        os.makedirs(root)

    plot = myplot(delay = 0.001)
    plot.init_plot(poses[:,0],poses[:,1],c='k',s=10)
    plot.xlabel('m')
    plot.ylabel('m')

    descptrs = np.array(list(descriptors.values()))
    #map_dptrs = np.array([descriptors[i] for i in map_idx])

    if record_gif == True:
        plot.record_gif(f'training_sim.gif')

    for a in anchor:
        query_desptr = np.expand_dims(descptrs[a],axis=0)
        sim = L2_np(query_desptr,descptrs,dim=1)
        c,s = color_similarity_on_map(sim,a)
        plot.update_plot(poses[:,0],poses[:,1],color = c , offset= 1, zoom=-1,scale=s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer.py")
    
    parser.add_argument(
      '--model', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the trained model.'
    )

    parser.add_argument(
      '--resume', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the trained model.'
    )

    parser.add_argument(
      '--dataset',
      type=str,
      required=False,
      default='kitti',
      help='Directory to get the trained model.'
    )

    parser.add_argument(
      '--sequence',
      type=str,
      required=False,
      default='00',
      help='Directory to get the trained model.'
    )

    parser.add_argument(
      '--modality',
      type=str,
      required=False,
      default='range',
      help='Directory to get the trained model.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    session_cfg_file = os.path.join('sessions', f'{FLAGS.dataset}.yaml')

    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    SESSION['val_loader']['data']['modality'] = FLAGS.modality
    
    # open arch config file
    cfg_file = os.path.join('dataloader','sensor-cfg.yaml')

                                        )
    
    record_gif = True
    loader  = orchard_loader
    poses   = loader.get_pose()
    anchor  = loader.anchors
    map_idx = loader.map_idx

    param = FLAGS.modality + '_param'
    SESSION['model']['type'] = FLAGS.model
    model_ = model.ModelWrapper(**SESSION['model'],loss= None, **SESSION[param])
    
    if FLAGS.resume != None:
        checkpoint = torch.load(FLAGS.resume)
        model_.load_state_dict(checkpoint['state_dict'])
    
    file_name = os.path.join('predictions',f'{FLAGS.dataset}-{FLAGS.sequence}-{FLAGS.modality}.npy')

    if not os.path.isfile(file_name):
        descriptors = generate_descriptors(model_,loader,device = 'cuda')
        torch.save(descriptors,file_name)    
    else:
        descriptors = torch.load(file_name)

    plot_sim_on_map(descriptors,anchor,map_idx,poses,record_gif=False)
    