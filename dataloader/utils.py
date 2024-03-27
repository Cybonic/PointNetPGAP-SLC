
import os
import numpy as np
from tqdm import tqdm


def get_files(target_dir):
    assert os.path.isdir(target_dir)
    files = np.array([f.split('.')[0] for f in os.listdir(target_dir) if f.endswith('.bin')])
    idx = np.argsort(files)
    fullfiles = np.array([os.path.join(target_dir,f) for f in os.listdir(target_dir)])
    return(files[idx],fullfiles[idx])


def gen_gt_constrained_by_rows(   poses,
                                retangle_rois, 
                                pos_range= 0.05, # Loop Threshold [m]
                                neg_range=10,
                                num_neg = 10,
                                num_pos = 10,
                                warmupitrs= 10, # Number of frames to ignore at the beguinning
                                roi       = 5, # Window):
                                anchor_range = 0
                    ):
    """
    Generate ground truth for loop closure detection    
    Args:
        poses (np.array): Array of poses
        retangle_rois (np.array): [[xmin,xmax,ymin,ymax],
                                    [xmin,xmax,ymin,ymax],
                                    ...
                                    [xmin,xmax,ymin,ymax]
                                    ]

        pos_range (float, optional): [description]. Defaults to 0.05.
        neg_range (int, optional): [description]. Defaults to 10.
        num_neg   (int, optional): [description]. Defaults to 10.
        num_pos   (int, optional): [description]. Defaults to 10.
        warmupitrs (int, optional): [description]. Defaults to 10.
        roi (int, optional): [description]. Defaults to 5.
    Returns:
        [type]: [description]
    """

    indices = np.array(range(poses.shape[0]-1))[warmupitrs:]
    anchors =   []
    positives = []
    negatives = []

    n_coord=  poses[0].shape[0]
    last_anchor_pose =poses[0,:].reshape((1,-1))
    all_idnices = np.array(range(poses.shape[0]-1))
    for i in indices:
    
        pose    = poses[i,:].reshape((1,-1))
        # Compute the Euclidean distance between the anchor point and the last anchor
        dist_meter  = np.sqrt(np.sum((pose -last_anchor_pose)**2,axis=1))
        # If the distance is greater than anchor_range, then the current pose is an anchor
        if dist_meter < anchor_range:
            continue
        
        _map_   = poses

        # Compute the Euclidean distance between the anchor point and the map
        dist_meter  = np.sqrt(np.sum((pose - _map_)**2,axis=1))
        
        exlude_window = np.arange(i-roi,i)
        # Remove window indices from the list of all indices
        pos_indices_of_interest = all_idnices[:i-roi].copy()
        neg_indices_of_interest = all_idnices

        last_anchor_pose = pose
        pos_dis_   = dist_meter[pos_indices_of_interest]
        neg_dis    = dist_meter[neg_indices_of_interest]
        
        # Select the points within the positive range
        pos_idx_in_range = np.where(pos_dis_ <= float(pos_range))[0]
        pos_idx_in_range = pos_indices_of_interest[pos_idx_in_range]

        # Select the points outside the negative range
        neg_idx_out_range = np.where(neg_dis >= float(neg_range))[0]
        neg_idx_out_range = neg_indices_of_interest[neg_idx_out_range]

        if len(pos_idx_in_range)>0:
            all_neg_poses = poses[neg_idx_out_range,:]
            all_pos_poses = poses[pos_idx_in_range,:]
            # Identify the row ID for each anchor, positive and negative point
            an_labels  = extract_points_in_rectangle_roi(pose,retangle_rois)
            pos_labels = extract_points_in_rectangle_roi(all_pos_poses,retangle_rois)
            neg_labels = extract_points_in_rectangle_roi(all_neg_poses,retangle_rois)

            pos_idx_in_same_row = []
            neg_idx_out_same_row = []

            if an_labels[0]== -1:
                continue
            
            # Anchor and positives must be in the same row
            pos_idx_in_same_row  = np.where(an_labels[0] == pos_labels)[0]
            # Negatives must be in a different row
            neg_idx_out_same_row = np.where(an_labels[0] != neg_labels)[0]

            if len(pos_idx_in_same_row) and len(neg_idx_out_same_row):

                pos_map_idx = np.array(pos_idx_in_range[pos_idx_in_same_row]) # Select the points in the same row
                seleced_pos_dist = dist_meter[pos_map_idx]
                min_sort = np.argsort(seleced_pos_dist)  # Sort by distance to the anchor point, nearest first
                
                if num_pos < 0: # Select all the points in the positive range
                    positives.append(pos_map_idx[min_sort]) # Select the nearest point
                else:
                    positives.append(pos_map_idx[min_sort][:num_pos]) # Select the nearest point

                neg_map_idx = np.array(neg_idx_out_range[neg_idx_out_same_row]) # Select the points in the same row
                seleced_neg_dist = dist_meter[neg_map_idx]
                min_sort = np.argsort(seleced_neg_dist)

                intervals = int(round((len(neg_map_idx))/(num_neg),0)) # compute the interval step size
                select_neg = np.arange(0,len(neg_map_idx),intervals)[:num_neg] # split the negatives in regular intervals groups
                negatives.append(neg_map_idx[min_sort][select_neg]) # Select the nearest point
                
                anchors.append(i)

    
    # Negatives


    return(anchors,positives,negatives)



def gen_ground_truth(   poses,           # Poses [N,3]
                        pos_range= 0.05, # Positive range Threshold [m]
                        num_pos =  10,   # Number of positives to be returned
                        warmupitrs= 10,  # Number of frames to ignore at the beguinning
                        roi       = 5,   # Window [in frames] arround the anchor frame to be ignored
                        **kwargs         # Additional arguments to ignore
                    ):
    """
    Generate ground truth for loop closure detection

    Args:
        poses (np.array): Array of poses
        pos_range (float, optional): [description]. Defaults to 0.05.
        neg_range (int, optional): [description]. Defaults to 10.
        num_neg   (int, optional): [description]. Defaults to 10.
        num_pos   (int, optional): [description]. Defaults to 10.
        warmupitrs (int, optional): [description]. Defaults to 10.
        roi (int, optional): [description]. Defaults to 5.
    Returns:
        [type]: [description]
    """

    assert roi<warmupitrs, "ROI must be smaller than warmupitrs"
    assert num_pos>0, "Number of positive must be greater than 0"
    
    indices = np.array(range(poses.shape[0]-1))
    
    ROI = indices[warmupitrs:]
    anchor   = []
    positive = []

    for i in tqdm(ROI,"Generating Ground Truth: Anchors and Positives"):
        _map_   = poses[:i-roi,:] # Get the points before the anchor point to build the map, points between i and  i-ROI are ignored
        pose    = poses[i,:].reshape((1,-1)) # Anchor frame
        # Compute the Euclidean distance between the anchor point and the map
        dist_meter  = np.sqrt(np.sum((pose -_map_)**2,axis=1))
        # Select the points in the positive range
        pos_idx = np.where(dist_meter < pos_range)[0]
        # If there are points in the positive range
        if len(pos_idx) >= num_pos:
            pos_dist = dist_meter[pos_idx]
            min_sort = np.argsort(pos_dist)
            
            if num_pos == -1: # Select all the points in the positive range
                pos_select = pos_idx[min_sort]
            else:           # Select the nearest points defined by num_pos
                pos_select = pos_idx[min_sort[:num_pos]]
            # Append the anchor and positive points indices
            positive.append(pos_select)
            anchor.append(i)

    return(anchor,positive)





def rotate_poses(xy,angle=-4 ):
    """
    Rotate the poses to match the image frame

    Args:
        xy (np.array): Array of poses
        angle (int, optional): Angle (in degrees) to aline the rows with the image frame. Defaults to -4.

    Returns:
        np.array: Array of rotated poses
    """


    import math
    xy = xy.copy().transpose() # Grid
    myx = np.mean(xy,axis=1).reshape(-1,1)

    #print(xy)
    xyy= xy - myx
    theta = math.radians(angle) # Align the map with the grid 
    rot_matrix = np.array([[math.cos(theta), -math.sin(theta),0],
                          [ math.sin(theta),  math.cos(theta),0],
                          [0,0,1]])
    
    new_xx = rot_matrix.dot(xyy) + myx

    return(new_xx.transpose())


def extract_points_in_rectangle_roi(points:np.ndarray,rois:np.ndarray) -> np.ndarray:
    """
    Return the points inside the rectangle

    Args:
        points (np.array): Array of points
        retangle edges (np.array): [[xmin,xmax,ymin,ymax],
                                    [xmin,xmax,ymin,ymax],
                                    ...
                                    [xmin,xmax,ymin,ymax]]
    Returns:
        np.array: row ids
        np.array: point indices
    """

    assert isinstance(points,np.ndarray),'Points should be a numpy array'
    assert isinstance(rois,np.ndarray),'ROI should be a numpy array'
    assert rois.shape[1]==4,'ROI should be a 2D array'
    assert rois.shape[0]>0,'ROI should be a 2D array'
    assert points.shape[0]>0,'Points should be a 2D array'


    labels = []
    row_id = -1 * np.ones(points.shape[0],np.int32)
    for i,roi in enumerate(rois):
        # 
        xmin = (points[:,0]>=roi[0])
        xmax = (points[:,0]<roi[1])
        xx = np.logical_and(xmin, xmax)

        ymin = (points[:,1]>=roi[2]) 
        ymax = (points[:,1]<roi[3])
        yy = np.logical_and(ymin, ymax)
        
        selected = np.logical_and(xx, yy)
        idx  = np.where(selected==True)[0]
        

        if len(idx)>0:
            row_id[idx] = i

    return row_id




def make_data_loader(root_dir,dataset,session,modality,max_points=50000):

    dataset = dataset.lower()
    assert dataset in ['kitti','orchard-uk','uk','pointnetvlad'],'Dataset Name does not exist!'

    
    #if dataset == 'kitti':
        # Kitti 
    from dataloader.kitti.kitti import KITTI as DATALOADER
    
    loader = DATALOADER( root = root_dir,
                    dataset = dataset,
                    modality = modality,
                    memory   = session['memory'],
                    train_loader  = session['train_loader'],
                    val_loader    = session['val_loader'],
                    max_points    = session['max_points']
                    )
    
    #elif dataset in ['orchards-uk','uk'] :

    #    from .ORCHARDS import ORCHARDS

    #    loader = ORCHARDS(  root    = root_dir,
    #                        modality = modality,
    #                        train_loader  = session['train_loader'],
    #                        test_loader   = session['val_loader'],
    #                        memory = session['memory'],
    #                        max_points    = session['max_points']
    #                        )
    
    
    #elif dataset == 'pointnetvlad':
        
    #    from .POINTNETVLAD import POINTNETVLAD
        
    #    loader = POINTNETVLAD(root       = session[root_dir],
    #                        train_loader = session['train_loader'],
    #                        val_loader   = session['val_loader'],
    #                        memory       = memory
    #                        )
    

    #elif dataset == 'fuberlin':
        
        #session['train_loader']['root'] =  session[root_dir]
    #    session['val_loader']['root'] =  session[root_dir]
    #    loader = FUBERLIN(
    #                        train_loader  = session['train_loader'],
    #                        val_loader    = session['val_loader'],
    #                        mode          = memory,
    #                        max_points = 50000
    #                        )
    
    return(loader)



    
