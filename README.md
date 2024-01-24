
## Installation 
Set up environment
- python 3.9.4
- Cuda 11.7
- pytorch 2.0.1
- open3d
- torchpack

Create conda environment with python:
```
conda create -n pr_env python=3.9.4
conda activate pr_env
```

Install Conda 11.7.0
```
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
```

Install Pytorch 2.0.1
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Testing installation
```
python "import torch; print("torch.cuda.is_available())"
```

Install sparse
```
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0
```

### Install other dependencies 

```
pip install -r requirements.txt
```





# DATASET


# Triplet Ground Truth: 

Pre-generated triplet files: 
A loop exists (ie anchor-positive pair) whenever  two samples from different revisits are whithin a range of 2 meters;   
For each anchor, exists: 
 - 1 positive, nearest neigbor within a range of 2 meters,
 - 20 negatives, selected from outside a range of 10 meters,

For the purpose of this work, 4 pre-defined triplet data where generated  and stored in the pickled files.
The files names incode information regarding the selection process of the data. 
E.g., the file "ground_truth_ar0.1m_nr10m_pr2m.pkle" comprises anchors (ar) are sperated by at least 0.1m.
the negatives where generated from outside a range of 10m and the positive was selected whithin a range of 2m.

The four predefined triplet data files are the following:
 - ground_truth_ar0.1m_nr10m_pr2m.pkle
 - ground_truth_ar0.5m_nr10m_pr2m.pkle
 - ground_truth_ar1m_nr10m_pr2m.pkle
 - ground_truth_ar5m_nr10m_pr2m.pkle

## Number of anchor positive pairs (ie loops):
The following information represents the number of loops that exist for each ground truth file. The number of loops also correspond to the number of training samples in each file. 

Orchards/aut22:\
AR5.0m: 45
AR1.0m: 252
AR0.5m: 495
AR0.1m: 1643

Orchards/sum22:
AR5.0m: 10
AR1.0m: 53
AR0.5m: 100
AR0.1m: 391

Orchards/june23:
AR5.0m: 54
AR1.0m: 268
AR0.5m: 512
AR0.1m: 1982

Strawberry/june23:
AR5.0m: 66
AR1.0m: 311
AR0.5m: 598
AR0.1m: 1942

greenhouse/e3:
AR5.0m: 14
AR1.0m: 73
AR0.5m: 117
AR0.1m: 242
