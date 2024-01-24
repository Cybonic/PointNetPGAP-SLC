
import os,sys
import numpy as np
# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory and add it to the Python path
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))

def read_poses(file):
    T_array = []
    for line in open(file):
        l = line.strip().split(' ')
        l_num = np.asarray([float(value) for value in l])
        T_array.append(l_num)
    return np.stack(T_array,axis=0)


# Read the point cloud from a PLY file


if __name__ == "__main__":
    target_dir = "/home/beast/Dropbox/SHARE/DATASET/uk/strawberry/june23/extracted"
    assert os.path.exists(target_dir)
    file = os.path.join(target_dir,"positions.txt")

    poses = read_poses(file)

    import numpy as np 

    d = 0
    for i in range(1,len(poses)):
        delta = poses[i-1][0:2] - poses[i][0:2]
        
        d += np.sqrt(np.sum(delta[:2]**2,axis=0))
    
    print("Total distance: ")
    print(d)
