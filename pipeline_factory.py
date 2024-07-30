
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan
from dataloader.datasets.loader import cross_validation


from networks.pipelines.PointNetVLAD import PointNetVLAD
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

    if pipeline_name.startswith('LOGG3D'):
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name.startswith('PointNetMAC'):
        from networks.pipelines.MACNet import PointNetMAC
        pipeline = PointNetMAC(output_dim=output_dim,feat_dim=feat_dim,num_points=num_points)
    elif pipeline_name.startswith('PointNetGeM'):
        from networks.pipelines.GeMNet import PointNetGeM
        pipeline = PointNetGeM(output_dim=output_dim,feat_dim=1024,num_points=num_points)
    elif pipeline_name.startswith('PointNetPGAP'):
        from networks.pipelines.PointNetPGAP import PointNetPGAP
        pipeline = PointNetPGAP(input_channels=3, output_channels=16, use_xyz=True, num_points=num_points)
    elif pipeline_name.startswith('PointNetVLAD'):
        pipeline = PointNetVLAD(use_tnet=True, output_dim=output_dim, num_points = num_points, feat_dim = 1024)
    elif pipeline_name.startswith('overlap_transformer'):
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)
    else:
        raise NotImplementedError(f"Network not implemented!: {pipeline_name}")

    loss = None
    if 'loss' in argv and argv['loss'] is not None:
        loss_type  = argv['loss']['type']
        loss_param = argv['loss']['args']

        loss = losses.__dict__[loss_type](**loss_param,device = device)

    print("*"*30)
    print(f'Loss: {loss}')
    print("*"*30)

    if pipeline_name.startswith('LOGG3D') or pipeline_name.startswith("SPV"):
        # Voxelized point cloud based model
        
        if pipeline_name.endswith('Loss'):
            # with SLC loss 
            model = contrastive.SparseModelWrapperLoss(pipeline,
                                                       loss = loss,
                                                       aux_loss = 'segment_loss',
                                                       device = device,
                                                       **argv['trainer'],
                                                       n_classes = argv['n_classes'],
                                                       loss_margin=0.5 if 'alpha' not in argv else argv['alpha'])
        else:
            # without SLC loss
            model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
        #model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
    
    elif pipeline_name.endswith('Loss'):
        # Point cloud based model with contrastive loss
        # with SLC loss
        model = contrastive.ModelWrapperLoss(pipeline,
                                             loss = loss,
                                             aux_loss = 'segment_loss',
                                             device = device,
                                             **argv['trainer'],
                                             n_classes = argv['n_classes'],
                                             loss_margin=0.5 if 'alpha' not in argv else argv['alpha'])
        
    elif pipeline_name.startswith('PoinNetPGAP'):
        from networks import regression
        # Point cloud based model         
        # No SLC loss
        model = regression.ModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
    else: 
        model = contrastive.ModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
        

    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)

    return model

# ==================================================================================================
# ======================================== DATALOADER FACTORY ======================================
# ==================================================================================================

def dataloader_handler(root_dir,network,
                       dataset,
                       val_set,
                       session,
                       pcl_norm=False,
                       model_evaluation='cross_validation',
                       **args):

    # Load the predefined data splits 
    datasplits = yaml.load(open("sessions/full_data_splits.yaml", 'r'),Loader=yaml.FullLoader)
    # Get the training and validation sequences based on VAL_SET
    #experiment = args['experiment']
    
    model_evaluation_exp = model_evaluation +'_'+ val_set if model_evaluation == 'cross_domain' else model_evaluation
    
    print(f"\n[INFO]Experiment: {model_evaluation_exp}")
    
    if 'cross_validation' in model_evaluation_exp:
        session['train_loader']['sequence'] = datasplits[model_evaluation_exp]['seq'][val_set] # Get the training sequences for val_set
        session['train_loader']['dataset']  = datasplits[model_evaluation_exp]['dataset']
        session['val_loader']['sequence'] = [val_set]
        session['val_loader']['dataset'] = datasplits[model_evaluation_exp]['dataset']
        
    elif model_evaluation_exp.startswith('cross_domain'):
        session['train_loader']['sequence'] = datasplits[model_evaluation_exp]['train']['seq'] # Get the training sequences for val_set
        session['train_loader']['dataset']  = datasplits[model_evaluation_exp]['train']['dataset']
        
        session['val_loader']['sequence'] = datasplits[model_evaluation_exp]['val']['seq'] 
        session['val_loader']['dataset'] = datasplits[model_evaluation_exp]['val']['dataset']
    else:
        raise NotImplementedError("Model Evaluation not implemented!")
        
    #sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)

    roi = None
    if 'roi' in args and args['roi'] > 0:
        roi = {}
        print(f"\nROI: {args['roi']}\n")
        roi['xmin'] = -args['roi']
        roi['xmax'] = args['roi']
        roi['ymin'] = -args['roi']
        roi['ymax'] = args['roi']

    # Select the modality based on the network
    if network.startswith('overlap_transformer'):
        # BEV based modality
        modality = BEVProjection(width=256,height=256,square_roi=roi)
            
    elif network.startswith('LOGG3D') or network.startswith("SPV"):
        # Get sparse (voxelized) point cloud based modality
        num_points=session['max_points']
        modality = SparseLaserScan(voxel_size=0.1,max_points=num_points, pcl_norm = False)
    
    elif network in ['PoinNetPGAP','PoinNetPGAPLoss','PointNetVLADLoss','PointNetMACLoss','PointNetVLAD'] or network.startswith("PointNet"):
        # Get point cloud based modality
        num_points = session['max_points']
        modality = Scan(max_points=num_points,square_roi=roi, pcl_norm=pcl_norm,clean_zeros=False)
    else:
        raise NotImplementedError("Modality not implemented!")

    # Select experiment type by default is cross_validation
    # model_evaluation = "cross_validation" # Default

    #if "model_evaluation" in session:
    #    model_evaluation = session['model_evaluation']

    print(f"\n[INFO]Model Evaluation: {model_evaluation}")

    if model_evaluation in ["cross_validation",'cross_domain']:
        loader = cross_validation(  root = root_dir,
                                    dataset = dataset,
                                    modality = modality,    
                                    memory   = session['memory'], # DISK or RAM
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
        
    else:
        raise NotImplementedError("Model Evaluation not implemented!")

    return loader
