
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan
from dataloader.kitti.kitti import cross_validation,split


from networks.pipelines.PointNetVLAD import PointNetVLAD
from networks.pipelines.LOGG3D import LOGG3D
from networks.pipelines import ORCHNet
from networks.pipelines.GeMNet import PointNetGeM,ResNet50GeM
from networks.pipelines.overlap_transformer import featureExtracter
from networks.pipelines.SPoCNet import PointNetSPoC,ResNet50SPoC
from networks.pipelines.MACNet import PointNetMAC,ResNet50MAC 
import yaml

from utils import loss as losses
from networks import contrastive

# ==================================================================================================
MODELS = ['LOGG3D',
          'PointNetVLAD',
          'ORCHNet_PointNet',
          'ORCHNet_ResNet50',
          'overlap_transformer',
          'ORCHNet']

# ==================================================================================================
# ======================================== PIPELINE FACTORY ========================================
# ==================================================================================================

def model_handler(pipeline_name, num_points=4096,output_dim=256,feat_dim=1024,device='cuda',**argv):
    """
    This function returns the model 
    
    Parmeters:
    ----------
    pipeline_name: str
        Name of the pipeline to be used
    num_points: int
        Number of points to be used as input
    output_dim: int
        Dimension of the output feature vector
    feat_dim: int
        Dimension of the hidden feature vector

    Returns:
    --------
    pipeline: object
        Pipeline object
    """
    
    print("\n**************************************************")
    print(f"Model: {pipeline_name}")
    print(f"N.points: {num_points}")
    print(f"Dpts: {output_dim}")
    print(f"Feat Dim: {feat_dim}")
    print("**************************************************\n")

    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD(use_tnet=True, output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif "ORCHNet" in pipeline_name:
        pipeline = ORCHNet.__dict__[pipeline_name](output_dim=output_dim, num_points=num_points,feat_dim = feat_dim)
    elif pipeline_name == "PointNetGeM":
        pipeline = PointNetGeM(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50GeM":    
        pipeline = ResNet50GeM(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == "PointNetSPoC":
        pipeline = PointNetSPoC(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50SPoC":    
        pipeline = ResNet50SPoC(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == "PointNetMAC":
        pipeline = PointNetMAC(output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name == "ResNet50MAC":
        pipeline = ResNet50MAC(output_dim=output_dim,feat_dim = 1024)
    elif pipeline_name == 'overlap_transformer':
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)
    else:
        raise NotImplementedError("Network not implemented!")

    loss = None
    if 'loss' in argv:
        loss_type  = argv['loss']['type']
        loss_param = argv['loss']['args']

        loss = losses.__dict__[loss_type](**loss_param,device = device)

    print("*"*30)
    print(f'Loss: {loss}')
    print("*"*30)

    if pipeline_name in ['LOGG3D'] or pipeline_name.startswith("spvcnn"):
        # Sparse model has a different wrapper, because of the splitting 
        model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['modelwrapper'])
    else:
        model = contrastive.ModelWrapper(pipeline,loss =loss,device = device, **argv['modelwrapper'])

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)

    return model

# ==================================================================================================
# ======================================== DATALOADER FACTORY ======================================
# ==================================================================================================

def dataloader_handler(root_dir,network,dataset,session,**args):

    assert dataset in ['kitti','orchard-uk','uk'],'Dataset Name does not exist!'

    sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)
    sensor_pram = sensor_pram[dataset]


    roi = None
    if 'roi' in args and args['roi'] > 0:
        roi = sensor_pram['square_roi']
        print(f"\nROI: {args['roi']}\n")
        roi['xmin'] = -args['roi']
        roi['xmax'] = args['roi']
        roi['ymin'] = -args['roi']
        roi['ymax'] = args['roi']

    if network in ['ResNet50_ORCHNet','overlap_transformer',"ResNet50GeM"] or network.startswith("ResNet50"):
        # These networks use proxy representation to encode the point clouds
        if session['modality'] == "bev" or network == "overlap_transformer":
            bev_pram = sensor_pram['bev']
            modality = BEVProjection(**bev_pram,square_roi=roi,aug_flag=session['aug'])
        elif session['modality'] == "spherical" or network != "overlap_transformer":
            modality = SphericalProjection(256,256,square_roi=roi,aug_flag=session['aug'])
            
    elif network in ['LOGG3D'] or network.startswith("spvcnn"):
        # Get sparse (voxelized) point cloud based modality
        num_points=session['max_points']
        output_dim=256
        modality = SparseLaserScan(voxel_size=0.05,max_points=num_points,
                                   aug_flag=session['aug'])
    
    elif network in ['PointNetVLAD','PointNet_ORCHNet',"PointNetGeM"] or network.startswith("PointNet"):
        # Get point cloud based modality
        num_points = session['max_points']
        output_dim=256
        modality = Scan(max_points=num_points,
                        aug_flag=session['aug'],square_roi=roi,pcl_norm = False)
    else:
        raise NotImplementedError("Modality not implemented!")

    dataset = dataset.lower()

    # Select experiment type by default is cross_validation
    model_evaluation = "cross_validation" # Default

    if "model_evaluation" in session:
        model_evaluation = session['model_evaluation']

    print(f"\n[INFO]Model Evaluation: {model_evaluation}")

    if model_evaluation == "cross_validation":
        loader = cross_validation( root = root_dir,
                                    dataset = dataset,
                                    modality = modality,
                                    memory   = session['memory'],
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
    elif model_evaluation == "split":
        loader = split( root = root_dir,
                                    dataset = dataset,
                                    modality = modality,
                                    memory   = session['memory'],
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
    else:
        raise NotImplementedError("Model Evaluation not implemented!")

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