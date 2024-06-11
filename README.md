# PointNetPGAP:  PointNetPGAP-SLC: A 3D LiDAR-based Place Recognition Approach with Segment-level Consistency Training for Mobile Robots in Horticulture

### Authors:  T. Barros, L. Garrote, P. Conde, M.J. Coombes, C. Liu, C. Premebida, U.J. Nunes

# Installation

You can install PointNetGAP locally in your machine.  We provide an complete installation guide for conda.


1. Create conda environment with python: ``` conda create -n pr_env python=3.9.4 ```

2. Activate conda environment
```  conda activate pr_env ```

3. Install cuda 11.7.0 ``` conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit    ```

4. Install Pytorch 2.0.1 ``` conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia ```

5. Testing installation ``` .... ```

6. Install sparse ``` pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0 ```

6.1 ```sudo apt-get install g++ ```

6.2 ```sudo apt-get install libsparsehash-dev ```

7. Install other dependencies  ``` pip install -r requirements.txt```


# Train

### Default training script
```
python script_train.py
```


### Costum training/testing
```
python train_knn.py  
        --network PointNetPGAPLoss # networl_name
        --train 1 # [1,0] [Train,Test] Train or test
        --dataset_root path/to/dataset/root # path to Dataset 
        --val_set 
        --memory RAM # [DISK, RAM] 
        --device cuda # Device
        --save_predictions path/to/dir # To save the predictions
        --epochs 200
        --max_points 10000
        --experiment experiment_name
        --feat_dim 16
        --eval_batch_size 15
        --mini_batch_size 1000
        --loss_alpha 0.5
```

# Testing on generated descriptors 

### Default training script:

In the ```script_eval.py```, edit:

```dataset_root = path/to/dataset/root```


```resume_root = path/to/descriptors```


Then, Run:
```
python script_eval.py
```

### Download Descriptors and Predictions [here](https://nas-greenbotics.isr.uc.pt/drive/d/s/yqEsJo2CzrFVr8lAQmRhSpftw2dBnIoh/B8IXnvGfsnqGC_BABb7n9qggaw4HhFGD-ZrhgM00gbgs)


## HORTO-3DLM Dataset

The HORTO-3DLM dataset comprises four sequences OJ22, OJ23, ON22, and SJ23;  Three sequences from orchards, namely from apples and cherries; and one sequence from strawberries;


### 3D Maps 
[Download HORTO-3DLM here](https://nas-greenbotics.isr.uc.pt/drive/d/s/x4eZ5aPL96blS0i7xNKIl0iJOtkdU7QR/h3YJb7wuqCZpV9NNxgeITnGTRsDJeVNY-a7eAQXUnGQs#file_id=799704328662196403)


