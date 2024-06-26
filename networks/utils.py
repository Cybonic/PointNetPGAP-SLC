import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = momentum

def IntermediateLayerGetter_(x,name,module,return_layers,hrnet_flag=False):

    out = {}
    if hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
        if name == 'transition1': # in transition1, you need to split the module to two streams first
            x = [trans(x) for trans in module]
        else: # all other transition is just an extra one stream split
            x.append(module(x[-1]))
    else: # other models (ex:resnet,mobilenet) are convolutions in series.
        x = module(x)

    if name in return_layers:
        out_name = return_layers[name]
        if name == 'stage4' and hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
            output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
            x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
            x = torch.cat([x[0], x1, x2, x3], dim=1)
        name = out_name
    
    out[name] = x  
    return(out)


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():

            if self.hrnet_flag and name.startswith('transition'): # if using hrnet, you need to take care of transition
                if name == 'transition1': # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]
                else: # all other transition is just an extra one stream split
                    x.append(module(x[-1]))
            else: # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if name == 'stage4' and self.hrnet_flag: # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = x[0].size(2), x[0].size(3)  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(x[1], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x2 = F.interpolate(x[2], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x3 = F.interpolate(x[3], size=(output_h, output_w), mode='bilinear', align_corners=False)
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out



def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class model_wrapper(nn.Module):
    def __init__(self,backbone,head):
        super(model_wrapper,self).__init__()
        self.backbone = backbone
        self.head = head 

    def forward(self,x):
        b = x.shape[0]
        s = x.shape[1]

        y = self.backbone(x)
        y = y['out']
        
        if len(y.shape)>3: # CNN-based output
            b,c,w,h = y.shape
            y = y.reshape(b,c,-1)
        
        if len(y.shape)<4: # Pointnet returns [batch x feature x samples]
            y = y.unsqueeze(dim=-1)
        z = self.head(y)

        return z
