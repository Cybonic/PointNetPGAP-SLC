
from networks.network_pipeline import get_pipeline
from dataloader import utils

from dataloader.projections import BEVProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan

def pipeline(network,dataset,session):


    if network == 'LOGG3D':
        # get nework
        num_points=session['max_points']
        output_dim=256
        modality = SparseLaserScan(max_points=num_points,
                                   aug_flag=session['aug'])
    elif network in ['PointNetVLAD','ORCHNet']:
        num_points = session['max_points']
        output_dim=256
        modality = Scan(max_points=num_points,
                        aug_flag=session['aug'])

    model_ = get_pipeline(network,num_points=num_points)

    loader = utils.load_dataset(dataset,session,modality)

    return model_,loader