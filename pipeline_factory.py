
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan



MODELS = ['LOGG3D',
          'PointNetVLAD',
          'ORCHNet_PointNet',
          'ORCHNet_ResNet50',
          'overlap_transformer',
          'ORCHNet']

def model_handler(pipeline_name, num_points=4096,output_dim=256):
    
    from networks.pipelines.PointNetVLAD import PointNetVLAD
    from networks.pipelines.LOGG3D import LOGG3D
    from networks.pipelines.ORCHNet import ORCHNet
    from networks.pipelines.GeMNet import PointNetGeM,ResNet50GeM
    from networks.pipelines.overlap_transformer import featureExtracter

    print("\n**************************************************")
    print(f"Model: {pipeline_name}")
    print(f"N.points: {num_points}")
    print(f"Dpts: {output_dim}")
    print("**************************************************\n")

    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD( use_tnet=True, output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name.endswith("ORCHNet"):
        if pipeline_name.startswith("PointNet"): # ['ORCHNet_PointNet','ORCHNet']:
            pipeline = ORCHNet('pointnet',output_dim=output_dim, num_points=num_points,feat_dim = 1024)
        elif pipeline_name.endswith("ResNet50"):# 'ORCHNet_ResNet50':
            pipeline = ORCHNet('resnet50',output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == "PointNetGeM":
        pipeline = PointNetGeM(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50GeM":    
        pipeline = ResNet50GeM(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == 'overlap_transformer':
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)
    else:
        raise NotImplementedError("Network not implemented!")   
    return pipeline


def dataloader_handler(root_dir,network,dataset,session):

    if network in ['ResNet50_ORCHNet','overlap_transformer',"ResNet50GeM"]:
        # These networks use proxy representation to encode the point clouds
        if session['modality'] == "bev":
            modality = BEVProjection(256,256)
        elif session['modality'] == "spherical":
            modality = SphericalProjection(256,256)

    elif network == 'LOGG3D':
        # Get sparse (voxelized) point cloud based modality
        num_points=session['max_points']
        output_dim=256
        modality = SparseLaserScan(voxel_size=0.05,max_points=num_points,
                                   aug_flag=session['aug'])
    
    elif network in ['PointNetVLAD','PointNet_ORCHNet',"PointNet_GeM"]:
        # Get point cloud based modality
        num_points = session['max_points']
        output_dim=256
        modality = Scan(max_points=num_points,
                        aug_flag=session['aug'])
    else:
        raise NotImplementedError("Network not implemented!")

    #loader = utils.make_data_loader(root_dir,dataset,session,modality)

    dataset = dataset.lower()
    assert dataset in ['kitti','orchard-uk','uk','pointnetvlad'],'Dataset Name does not exist!'

    from dataloader.kitti.kitti import KITTI as DATALOADER
    
    loader = DATALOADER( root = root_dir,
                    dataset = dataset,
                    modality = modality,
                    memory   = session['memory'],
                    train_loader  = session['train_loader'],
                    val_loader    = session['val_loader'],
                    max_points    = session['max_points']
                    )

    return loader



if __name__=="__main__":
    import yaml,os
    
    dataset = 'kitti'
    session_cfg_file = os.path.join('sessions', dataset.lower() + '.yaml')
    SESSION = yaml.safe_load(open(session_cfg_file, 'r'))
    Model,dataloader = pipeline('LOGG3D','kitti',SESSION)
    print(Model)
    print(dataloader)
    
    #assert str(Model)=="LO"