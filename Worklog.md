

# RUNNING MODEL
  ## LoGG3D-Net
    - conda activate orchnetv2
  ## wrning environment
    - conda base



# ISR PC PARAMETERS:
 Training param:
  - minibatch 20
  Testing param:
  - batchsize: 10

 

# SCAN-CONTEXT
  parameters: 
    https://gisbi-kim.github.io/publications/gkim-2018-iros.pdf
    Ns = 60, Nr = 20, and Lmax = 80 m. 
    That is, each sector has a 6◦ resolution and each ring has a 4 m gap. The number of bins of Z-projection is set as 100. 


# TODO:
[] ORCHNet: working on Kitti; 
[] Scancontext: generated the descriptors and performance.

[] Generate Descriptors for scancontext:
  7-03-2023: Code working to generate descriptors; 

[] Add feature to Knn-eval to load descriptors instead of generating the descriptors;
 7-03-2023: Code working to load previously saved descriptors

[] Finetune Scancontext

# WOKLOG 
## Training on pointnetvlad(oxford):
28/03/2023: Trained PointnetVlad model on the dataset, using the same parameters (ie f_dim=64,max_point: 1200, out_dim:256), but the model was not able to train. From the data(training), both dap and dan are converging to zero; which is not what is expected. I speculate that is due to a small amount of input points. 

## Tunning Models: 
  ORCHNET is working on the Kitti dataset; but it has the same behavior then on the orchard dataset, the d(a,p) and d(a,n) values are to small,  converging both to zero, Quadruplet Loss,   To solve this issue I did the following:
  - reduced both margin values of the QuadrupletLoss from 0.5 to 0.01;
    14/03/2023: no improvement observed
  - added a normal kernel function to the L2 Loss: 
    14/03/2023: at some point the loss diverged (ie  )
