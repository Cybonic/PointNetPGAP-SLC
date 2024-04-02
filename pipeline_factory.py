
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan
from dataloader.horto3dlm.loader import cross_validation


from networks.pipelines.PointNetVLAD import PointNetVLAD
from networks.pipelines.PointNetGAP import PointNetGAP
from networks.pipelines.LOGG3D import LOGG3D

from networks.pipelines.overlap_transformer import featureExtracter
import yaml

from utils import loss as losses
from networks import contrastive

# ==================================================================================================
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
    print("**************************************************\n")

    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD(use_tnet=True, output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name in ['PointNetGAP','PointNetGAPLoss']:
        pipeline = PointNetGAP(use_tnet=False, output_dim=output_dim, num_points = num_points, feat_dim = feat_dim)
    elif pipeline_name == 'overlap_transformer':
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)
    else:
        raise NotImplementedError("Network not implemented!")

    loss = None
    if 'loss' in argv and argv['loss'] is not None:
        loss_type  = argv['loss']['type']
        loss_param = argv['loss']['args']

        loss = losses.__dict__[loss_type](**loss_param,device = device)

    print("*"*30)
    print(f'Loss: {loss}')
    print("*"*30)

    if pipeline_name in ['LOGG3D'] or pipeline_name.startswith("SPV"):
        model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
    elif pipeline_name in ['PointNetGAPLoss']:
        model = contrastive.ModelWrapperLoss(pipeline,loss = loss,device = device,**argv['trainer'])
    else: 
        model = contrastive.ModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
        

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)

    return model

# ==================================================================================================
# ======================================== DATALOADER FACTORY ======================================
# ==================================================================================================

def dataloader_handler(root_dir,network,dataset,val_set,session,pcl_norm=False,**args):

    # Load the predefined data splits 
    datasplits = yaml.load(open("sessions/data_splits.yaml", 'r'),Loader=yaml.FullLoader)
    # Get the training and validation sequences based on VAL_SET
    session['train_loader']['sequence'] = datasplits['cross_validation'][val_set] # Get the training sequences for val_set
    session['val_loader']['sequence']   = [val_set]
    
    sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)

    roi = None
    if 'roi' in args and args['roi'] > 0:
        roi = {}
        print(f"\nROI: {args['roi']}\n")
        roi['xmin'] = -args['roi']
        roi['xmax'] = args['roi']
        roi['ymin'] = -args['roi']
        roi['ymax'] = args['roi']

    if network in ['overlap_transformer']:
        # These networks use proxy representation to encode the point clouds
        if session['modality'] == "bev" or network == "overlap_transformer":
            modality = BEVProjection(width=256,height=256,square_roi=roi)
        elif session['modality'] == "spherical" or network != "overlap_transformer":
            modality = SphericalProjection(256,256,square_roi=roi)
            
    elif network in ['LOGG3D'] or network.startswith("SPV"):
        # Get sparse (voxelized) point cloud based modality
        num_points=session['max_points']
        modality = SparseLaserScan(voxel_size=0.1,max_points=num_points, pcl_norm = False)
    
    elif network in ['PointNetVLAD','PointNetGAP','PointNetGAPLoss']:
        # Get point cloud based modality
        num_points = session['max_points']
        modality = Scan(max_points=num_points,square_roi=roi, pcl_norm=pcl_norm,clean_zeros=False)
    else:
        raise NotImplementedError("Modality not implemented!")

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
        
    else:
        raise NotImplementedError("Model Evaluation not implemented!")

    return loader
