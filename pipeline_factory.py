
# This file contains the factory function for the pipeline and dataloader
from dataloader.projections import BEVProjection,SphericalProjection
from dataloader.sparselaserscan import SparseLaserScan
from dataloader.laserscan import Scan
from dataloader.datasets.loader import cross_validation
from dataloader.hortov2.loader import validation_only

from networks.pipelines.PointNetVLAD import PointNetVLAD
from networks.pipelines.LOGG3D import LOGG3D
from networks.pipelines.SPVSoAP3D import SPVSoAP3D
from networks.pipelines.SPVSoAP3D import PointNetSoAP3D

from networks.pipelines.overlap_transformer import featureExtracter
import yaml

from utils import loss as losses
from networks import contrastive

# ==================================================================================================
# ==================================================================================================
# ======================================== PIPELINE FACTORY ========================================
# ==================================================================================================

def model_handler(network,device='cuda',**argv):
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
    
    architecture = network['architecture']
    output_dim   = network['output_dim']
    wrapper = network['wrapper']
    
    print("\n**************************************************")
    print(f"Model: {architecture}")
    print(f"Dpts: {output_dim}")
    print(f'Device: {device}')
    print(f'Wrapper: {wrapper}')
    print("**************************************************\n")

    if architecture.startswith('LOGG3D'):
        pipeline = LOGG3D(output_dim=output_dim)
    elif architecture.startswith('PointNetSoAP3D'):
        pipeline = PointNetSoAP3D(feat_dim=16)
    elif architecture.startswith('SPVSoAP3D'):
        pipeline = SPVSoAP3D(output_dim=output_dim,
                           local_feat_dim=16,
                           do_fc  = True, # use fully connected layer
                           do_epn = False, # use spectral power-norm
                           do_log = True, # use log
                           do_pn  = True, # use power-norm 
                           do_pnl = True, # trainable pwer-norm param (during training)
                           pres   = 0.1, # voxelization 
                           vres   = 0.1, # voxelization
                           )
    elif architecture.startswith('PointNetMAC'):
        from networks.pipelines.MACNet import PointNetMAC
        pipeline = PointNetMAC(output_dim=output_dim,feat_dim=1024)
    elif architecture.startswith('ResNet50MAC'):
        from networks.pipelines.MACNet import ResNet50MAC
        pipeline = ResNet50MAC(output_dim=output_dim,feat_dim=2048)
    elif architecture.startswith('SPVMAC'):
        from networks.pipelines.MACNet import SPVMAC
        pipeline = SPVMAC(output_dim=output_dim,feat_dim=16)
    elif architecture.startswith('PointNetGeM'):
        from networks.pipelines.GeMNet import PointNetGeM
        pipeline = PointNetGeM(output_dim=output_dim,feat_dim=1024)
    elif architecture.startswith('ResNet50GeM'):
        from networks.pipelines.GeMNet import ResNet50GeM
        pipeline = ResNet50GeM(output_dim=output_dim,feat_dim=2048)
    elif architecture.startswith('SPVGeM'):
        from networks.pipelines.GeMNet import SPVGeM
        pipeline = SPVGeM(output_dim=output_dim,feat_dim=16)
    elif architecture.startswith('PointNetVLAD'):
        pipeline = PointNetVLAD(use_tnet=True, output_dim=output_dim,feat_dim = 1024)
    elif architecture.startswith('SPVVLAD'):
        from networks.pipelines.PointNetVLAD import SPVVLAD
        pipeline = SPVVLAD(use_tnet=True, output_dim=output_dim, feat_dim = 16)
    elif architecture.startswith('ResNet50VLAD'):
        from networks.pipelines.PointNetVLAD import ResNet50VLAD
        pipeline = ResNet50VLAD(use_tnet=True, output_dim=output_dim, feat_dim = 2048)
    elif architecture.startswith('PointNetPGAP'):
        from networks.pipelines.PointNetPGAP import PointNetPGAP
        pipeline = PointNetPGAP(input_channels=3, output_channels=16, use_xyz=True)
    elif architecture.startswith('overlap_transformer'):
        pipeline = featureExtracter(channels=3,height=256, width=256, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024)
    else:
        raise NotImplementedError(f"Network not implemented!: {architecture}")

    loss = None
    if 'loss' in argv and argv['loss'] is not None:
        loss_type  = argv['loss']['type']
        loss_param = argv['loss']['args']

        loss = losses.__dict__[loss_type](**loss_param,device = device)

    print("*"*30)
    print(f'Loss: {loss}')
    print("*"*30)
    
    ## Wrapper to handle triplets 
    ## ************************************************************************
    
    ## SPARSE
    if architecture.startswith('LOGG3D') or architecture.startswith("SPV"):
        # Voxelized point cloud based model
        
        if architecture.endswith('Loss'):
            # with SLC loss 
            model = contrastive.SparseModelWrapperLoss(pipeline,
                                                       loss = loss,
                                                       aux_loss = 'segment_loss',
                                                       device = device,
                                                       **argv['run'],
                                                       n_classes = argv['n_classes'],
                                                       loss_margin=0.5 if 'alpha' not in argv else argv['alpha'])
        else:
            # without SLC loss
            model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['run'])
        #model = contrastive.SparseModelWrapper(pipeline,loss = loss,device = device,**argv['trainer'])
    
    elif architecture.endswith('Loss'):
        # Point cloud based model with contrastive loss
        # with SLC loss
        model = contrastive.ModelWrapperLoss(pipeline,
                                             loss = loss,
                                             aux_loss = 'segment_loss',
                                             device = device,
                                             **argv['run'],
                                             n_classes = argv['n_classes'],
                                             loss_margin=0.5 if 'alpha' not in argv else argv['alpha'])
        
    elif architecture.startswith('PointNetPGAP'):
        from networks import regression
        # Point cloud based model         
        # No SLC loss
        model = regression.ModelWrapper(pipeline,loss = loss,device = device,**wrapper)
    else:
        model = contrastive.ModelWrapper(pipeline,loss = loss,device = device,**wrapper)


    print("*"*30)
    print("Model: %s" %(str(model)))
    print("*"*30)

    return model

# ==================================================================================================
# ======================================== DATALOADER FACTORY ======================================
# ==================================================================================================

def dataloader_handler(network,
                       val_loader,
                       train_loader,
                       eval_protocol,
                       **args):

     # Load the predefined data splits 
    #datasplits = yaml.load(open("sessions/full_data_splits.yaml", 'r'),Loader=yaml.FullLoader)
    # Get the training and validation sequences based on VAL_SET
    #experiment = args['experiment']
    val_set = val_loader['dataset']['seq']
    # verify if sequence val is list our str
    if not isinstance(val_set, list):
        val_set = [val_set]
        
        
    input_format = network['input']
    architecture = network['architecture']
    num_points=input_format['num_points']
    
    model_evaluation_exp = eval_protocol +'_'+ val_set[0] # if eval_protocol == 'cross_domain' else eval_protocol
    
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
    
    elif eval_protocol == 'val_only':
        # print evaluation information
        print(f"\n[INFO]Evaluation: {model_evaluation_exp}")
        print(f"[INFO]Validation Dataset: {val_loader['dataset']['path']}")
        print(f"[INFO]Validation Sequence: {val_loader['dataset']['seq']}")
    else:
        raise NotImplementedError("Evaluation protocol not implemented!")
        
    #sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)


    # Input Data
    
    roi = None
    if 'roi' in input_format and input_format['roi'] > 0:
        roi = {}
        print(f"\nROI: {input_format['roi']}\n")
        roi['xmin'] = -input_format['roi']
        roi['xmax'] = input_format['roi']
        roi['ymin'] = -input_format['roi']
        roi['ymax'] = input_format['roi']


    # Select the modality based on the network
    if architecture.startswith('overlap_transformer') or architecture.startswith("ResNet"):
        # BEV based modality
        modality = BEVProjection(width=256,
                                 height=256,
                                 square_roi=roi)
            
    elif architecture.startswith('LOGG3D') or architecture.startswith("SPV"):
        # Get sparse (voxelized) point cloud based modality
        
        modality = SparseLaserScan(voxel_size=0.1,
                                    max_points=num_points,
                                    pcl_norm = False)
    
    elif architecture in ['PoinNetPGAP','PoinNetPGAPLoss','PointNetVLADLoss','PointNetMACLoss','PointNetVLAD'] or architecture.startswith("PointNet"):
        # Get point cloud based modality
        #num_points = input_format['max_points']
        modality = Scan(max_points=num_points,
                        square_roi=roi,
                        pcl_norm=False,
                        clean_zeros=False)
    else:
        raise NotImplementedError("Modality not implemented!")


    # Data Loader 

    print(f"\n[INFO]Model Evaluation: {eval_protocol}")

    if eval_protocol in ["cross_validation",'cross_domain']:
        loader = cross_validation(  root = root_dir,
                                    dataset = dataset,
                                    modality = modality,    
                                    memory        = session['memory'], # DISK or RAM
                                    train_loader  = session['train_loader'],
                                    val_loader    = session['val_loader'],
                                    max_points    = session['max_points']
                                    )
    elif eval_protocol == 'val_only':

        loader = validation_only( **val_loader,modality=modality)

    else:
        raise NotImplementedError("Model Evaluation not implemented!")

    return loader
