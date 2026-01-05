import numpy as np
import os

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

 