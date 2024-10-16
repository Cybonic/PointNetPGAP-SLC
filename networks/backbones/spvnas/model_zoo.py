import os
import sys
import torch
from torchpack import distributed as dist
sys.path.append(os.path.join(os.path.dirname(__file__)))
from networks.backbones.spvnas.core.models.spvcnn import SPVCNN

__all__ = ['spvcnn']


def spvcnn(output_dim=16,pres=0.05,vres=0.05,cr=0.64):

    model = SPVCNN(
        num_classes=output_dim,
        cr=cr,
        pres=pres,
        vres=vres
    ).to('cuda:%d' % dist.local_rank() if torch.cuda.is_available() else 'cpu')

    return model
