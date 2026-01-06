import numpy as np
import os

def elevate_along_path(positions, max_elevation=5.0):
    """
    Elevates positions along the path by gradually increasing z values.
    Creates a smooth elevation profile from start to end of the path.
    
    Args:
        positions: nx3 array of 3D positions
        max_elevation: Maximum z elevation to add (in meters)
        
    Returns:
        Elevated positions with z values increased along the path
    """
    positions_elevated = positions.copy()
    n_points = len(positions)
    
    # Create linear elevation gradient along the path
    elevation_gradient = np.linspace(0, max_elevation, n_points)
    
    # Add elevation to z-axis
    positions_elevated[:, 2] += elevation_gradient
    
    return positions_elevated


def elevate_along_path_smooth(positions, max_elevation=5.0, smoothness=2.0):
    """
    Elevates positions along the path with a smooth curve (sine wave).
    
    Args:
        positions: nx3 array of 3D positions
        max_elevation: Maximum z elevation to add (in meters)
        smoothness: Controls smoothness (higher = smoother but less variation)
        
    Returns:
        Elevated positions with smoothly curved z values
    """
    positions_elevated = positions.copy()
    n_points = len(positions)
    
    # Create smooth sine-based elevation
    t = np.linspace(0, smoothness * np.pi, n_points)
    elevation_gradient = max_elevation * (np.sin(t) + 1) / 2
    
    # Add elevation to z-axis
    positions_elevated[:, 2] += elevation_gradient
    
    return positions_elevated


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


def aligned_path(positions):
    """
    Aligns a 3D path with the x-y axes by detecting the longest straight line segment.
    Only rotates in the x-y plane, leaves z unchanged.
    
    Args:
        positions: nx3 array of 3D positions
        
    Returns:
        Aligned positions with longest straight line aligned to x-axis
    """
    # Translate to origin
    positions_aligned = positions - positions[0]
    
    # Find longest straight line segment
    max_length = 0
    best_angle = 0
    window_size = min(50, len(positions) // 4)  # Use a window to find straight segments
    
    for i in range(len(positions) - window_size):
        delta = positions_aligned[i + window_size] - positions_aligned[i]
        length = np.sqrt(delta[0]**2 + delta[1]**2)
        
        if length > max_length:
            max_length = length
            best_angle = np.arctan2(delta[1], delta[0])
    
    # Rotate so longest segment aligns with x-axis
    rot2d = np.array([[np.cos(-best_angle), -np.sin(-best_angle)],
                      [np.sin(-best_angle),  np.cos(-best_angle)]])
    positions_xy = positions_aligned[:, :2] @ rot2d.T
    
    # Keep z unchanged
    positions_rot = np.hstack([positions_xy, positions_aligned[:, 2:3]])
    return positions_rot


