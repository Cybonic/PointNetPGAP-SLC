

## Set up environment
- python 3.9.4
- Cuda 11.7
- pytorch 2.0.1
- open3d
- torchpack

## Create conda environment with python:
```
conda create -n pr_env python=3.9.4
conda activate pr_env
```
## Install Conda 11.7.0
```
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```
## Install Pytorch 2.0.1
```
pip install torch torchvision torchaudio
import torch
torch.cuda.is_available()
```
## Install sparse
```
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```