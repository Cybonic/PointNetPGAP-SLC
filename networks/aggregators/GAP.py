#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn


def _l2norm(x):
        x = nn.functional.normalize(x, p=2, dim=-1)
        return x

class GAP(nn.Module):
    def __init__(self, outdim=256,**argv):
        super().__init__()
        self.fc = nn.LazyLinear(outdim)

    def forward(self, x):
        # Return (batch_size, n_features) tensor
        x = x.view(x.shape[0],x.shape[1],-1)
        x = self.fc(torch.mean(x, dim=-1, keepdim=False)) # Return (batch_size, n_features) tensor
        return _l2norm(x)

