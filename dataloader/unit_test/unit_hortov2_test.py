"""Unit tests for the dataset loading and processing functions.
"""

# Add global path to sys path 
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from PointNetGAP.dataloader.hortov2.dataset import load_from_csv,file_structure, label_color_rgb, generate_label_colors
from PointNetGAP.dataloader.hortov2.utils import aligned_path, elevate_along_path, elevate_along_path_smooth # Linear elevation

COLORS = generate_label_colors(200)
ROOT_DIR = os.path.abspath("/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel")
SEQs = ["PCD_EASY",
        "PCD_Easy_DARK",
        "PCD_MED",
        "PCD_RAS_EASY"]

def test_LOAD_FROM_CSV():
    print("\n")
    print("Testing load_from_csv")
    for seq in SEQs:
        # Remove "pcd" from sequence name
        file_seq = seq.replace("PCD_", "")
        seq_dir = os.path.join(ROOT_DIR, seq)
        pose_csv_file = os.path.join(seq_dir,"path_{}".format(file_seq).lower() + ".csv")

        print(f"Loading test data from: {pose_csv_file}")
        test_data = load_from_csv(pose_csv_file)
        assert not test_data.empty, "Test data is empty"
        assert 'timestamp' in test_data.columns, "Missing timestamp column"
        assert 'ID' in test_data.columns, "Missing ID column"

    print("All tests for load_from_csv passed!")
    


def test_file_structure():

    for seq in SEQs:
        seq_dir = os.path.join(ROOT_DIR, seq)
        print(f"Testing file_structure on: {seq}")
        fs = file_structure(ROOT_DIR,seq, verbose=True)
        # Check target_dir
        assert fs._get_target_dir() == seq_dir
        # Check pose file loaded
        positions = fs._get_positions_()
        assert positions is not None and len(positions) > 0, "Pose file is empty or not loaded"
        # Check point cloud files
        pcl_files = fs._get_point_cloud_files_()
        assert isinstance(pcl_files, list) or isinstance(pcl_files, np.ndarray), "Point cloud files should be a list or ndarray"
        assert len(pcl_files) > 0, "No point cloud files found"

        # Plot path, colored with the label
        import matplotlib.pyplot as plt

        aligned_positions = aligned_path(positions)

        # Linear elevation
        elevated_positions = elevate_along_path(positions, max_elevation=10.0)

        # Or smooth elevation
        elevated_positions = elevate_along_path_smooth(positions, max_elevation=10.0, smoothness=1.0)

        # get label distribution
        label_distribution = fs._get_labels()
        print(f"Label distribution for {seq}: {np.unique(aligned_positions, return_counts=True)}")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(elevated_positions[:, 0], elevated_positions[:, 1], 
                   c=[COLORS[label] for label in fs._get_labels()], cmap='tab10')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Path for sequence: {seq}")
        ax.set_aspect('equal')  # Equal aspect ratio (square)
        # Or use: ax.set_aspect('auto')  # Default aspect ratio
        plt.legend()
        plt.show()


if __name__ == '__main__':
    # Load test data
    print("*"*10)
    print("\n")
    test_LOAD_FROM_CSV()

    print("*"*10)
    print("\n")
    
    test_file_structure()


    print("All tests passed!")
