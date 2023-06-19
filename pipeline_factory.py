
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
        modality = SparseLaserScan(voxel_size=0.05,max_points=num_points,
                                   aug_flag=session['aug'])
        
    elif network in ['PointNetVLAD','ORCHNet']:
        num_points = session['max_points']
        output_dim=256
        modality = Scan(max_points=num_points,
                        aug_flag=session['aug'])

    model_ = get_pipeline(network,num_points=num_points)

    loader = utils.make_data_loader(dataset,session,modality)

    return model_,loader


if __name__=="__main__":
    import yaml,os
    
    dataset = 'kitti'
    session_cfg_file = os.path.join('sessions', dataset.lower() + '.yaml')
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    Model,dataloader = pipeline('LOGG3D','kitti',SESSION)
    print(Model)
    print(dataloader)
    
    #assert str(Model)=="LO"