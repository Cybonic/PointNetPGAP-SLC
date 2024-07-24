import numpy as np
import os


def load_gps_to_RAM(file:str,local_frame=False)->np.ndarray:
    """ Loading the gps file to RAM.
    The gps file as the default structure of KITTI gps.txt file:
    Each line contains the following values separated by spaces:
    3D location (latitude, longitude, altitude)

    Args:
        file (str): default gps.txt file

    Returns:
        positions (np.array): array of positions (x,y,z) of the path (N,3)
        Note: N is the number of data points, No orientation is provided
    """
    assert os.path.isfile(file),"GPS file does not exist: " + file
    pose_array = []
    for line in open(file):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        if len(values) < 3:
            values = np.append(values,np.zeros(3-len(values)))
        elif len(values) > 3:
            values = values[:3]
        pose_array.append(values)

    pose_array = np.array(pose_array)
    return(pose_array)



def load_pose_to_RAM(file:str)->np.ndarray: 
    """ Loading the poses file to RAM. 
    The pose file as the default structure of KITTI poses.txt file:
    In each line, there are 16 values, which are the elements of a 4x4 Transformation matrix
    
    The elements are separated by spaces, and the values are in row-major order.

    T = [R11 R12 R13 tx   ->  [R11,R12,R13,tx,R21,R22,R23,ty,R31,R32,R33,tz,0,0,0,1]
         R21 R22 R23 ty
         R31 R32 R33 tz
         0   0   0   1 ]

    Args:
        file (str): default poses.txt file

    Returns:
        positions (np.array): array of positions (x,y,z) of the path (N,3)
        Note: N is the number of data points, No orientation is provided
    """
    assert os.path.isfile(file),"pose file does not exist: " + file
    pose_array = []
    for line in open(file):
        values_str = line.split(' ')
        values = np.array([float(v) for v in values_str])
        if len(values) < 16:
            # concatenate unite vector at the end
            values = np.append(values,[0,0,0,1])
            #values = np.append(values,np.zeros(16-len(values)))
        coordinates = np.array(values).reshape(4,4)
        pose_array.append(coordinates[:3,3])
        
    pose_array = np.array(pose_array)   
    return(pose_array)




def load_positions(file):
    if  "poses" in file:
        poses = load_pose_to_RAM(file)
    elif "gps" in file or "positions" in file:
        poses = load_gps_to_RAM(file)
    else:
        raise Exception("Invalid pose data source")
    
    return poses



def save_positions_KITTI_format(path:str,data:np.ndarray):
    """
    Save positions in KITTI format

    Parameters
    ----------
    file : str
        File name
    data : np.array 
        Array nx3 of positions 
    """

    assert isinstance(data,np.ndarray), "Data must be a numpy array"
    assert data.shape[1] == 3, "Data must be a nx3 array"
    assert os.path.isdir(path), "Path must be a directory"
    file = os.path.join(path,'positions.txt')
    fd = open(file,'w')
    for i in range(data.shape[0]):
        line = " ".join([str(x) for x in data[i]])
        line += "\n"
        fd.write(line)
    fd.close()

    print("[INF] Saved positions to: %s"% file)

 