
# from .backbone import resnet,pointnet
 # from .multihead import GeM,SPoC,MAC
from .utils import IntermediateLayerGetter,model_wrapper
from .netvlad import NetVLADLoupe as VLADHead

def _place_resnet(name, backbone_name, output_dim, output_stride, max_samples,pretrained_backbone,**argv):

    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
    else:
        replace_stride_with_dilation=[False, False, True]
    # Backbone
    in_ch = argv.pop('in_channels')
    backbone = resnet.__dict__[backbone_name](
                                    pretrained=pretrained_backbone,
                                    replace_stride_with_dilation=replace_stride_with_dilation, 
                                    in_channels = in_ch)
    
    inplanes = 2048
    return_layers = {'layer4': 'out'}
    # Attention
    if name == 'VLAD':
        AggHead = VLADHead(in_dim=inplanes,out_dim=output_dim,max_samples=2048) # the number of samples are the output of the CNN
    elif name == 'GeM':
        AggHead = GeM(outdim=output_dim)
    elif name == 'SPoC':
        AggHead = SPoC(outdim=output_dim)
    elif name.endswith('MAC'):
        AggHead = MAC(outdim=output_dim)

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    # Global Features
    model = model_wrapper(backbone, AggHead)
    return model


def _place_pointnet(name, backbone, output_dim,max_samples,**argv):
    
    inplanes = 1024
    
    backbone = pointnet.PointNet_features(dim_k=inplanes,use_tnet=argv['use_tnet'], scale=2)

    # Attention
    #if name == 'AttVLAD':
    #    classifier = AttVLADHead(in_dim=inplanes,out_dim=output_dim, max_samples=max_samples)
    if name.endswith('VLAD'):
        AggHead = VLADHead(in_dim=inplanes,out_dim=output_dim,max_samples=max_samples)
    elif name.endswith('GeM'):
        AggHead = GeM(outdim=output_dim)
    elif name.endswith('SPoC'):
        AggHead = SPoC(outdim=output_dim)
    elif name.endswith('MAC'):
        AggHead = MAC(outdim=output_dim)

    # Global Features
    model = model_wrapper(backbone, AggHead)
    return model



def _load_model(arch_type, backbone, output_dim,output_stride, pretrained_backbone,in_channels,**argv):

    if backbone == 'pointnet':
        model = _place_pointnet(arch_type, backbone, output_dim,**argv)

    elif backbone.startswith('resnet'):
        model = _place_resnet(arch_type, backbone, output_dim, output_stride= output_stride, pretrained_backbone=pretrained_backbone, in_channels=in_channels,**argv)
    
    else:
        raise NotImplementedError
    return model


# ================================================================
# PointNet
# ================================================================

def MAC_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('GeM', 'pointnet', output_dim,**argv)

def GeM_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('GeM', 'pointnet', output_dim,**argv)

def SPoC_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('SPoC', 'pointnet', output_dim,**argv)

def VLAD_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('VLAD', 'pointnet', output_dim,**argv)

def AttVLAD_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('AttVLAD', 'pointnet', output_dim,**argv)

def AttSPoC_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('AttSPoC', 'pointnet', output_dim,**argv)

def AttGeM_pointnet(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('AttGeM', 'pointnet', output_dim,**argv)

# ================================================================
# ResNet50
# ================================================================
def SPoC_resnet50(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('SPoC', 'resnet50', output_dim,**argv)

def GeM_resnet50(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('GeM', 'resnet50', output_dim,**argv)

def MAC_resnet50(output_dim=128,**argv):
    # Pretrained model has to be False, because there is no pretrained model available
    return _load_model('MAC', 'resnet50', output_dim,**argv)

def AttVLAD_resnet50( output_dim=128, **argv):
    return _load_model('AttVLAD', 'resnet50', output_dim,**argv)

def VLAD_resnet50( output_dim=128,**argv):
    return _load_model('VLAD', 'resnet50', output_dim,**argv)

# ================================================================
# ORCHNet
# ================================================================
from .pipelines.orchnet import ORCHNet

def ORCHNet_resnet50(**argv):
    return ORCHNet('resnet50', **argv)

def ORCHNet_pointnet(**argv):
    return ORCHNet('pointnet',**argv)



# ================================================================
# SCANCONTEXT
# ================================================================
from .scancontext.scancontext import SCANCONTEXT

def scancontext(**argv):
    return SCANCONTEXT(**argv)
