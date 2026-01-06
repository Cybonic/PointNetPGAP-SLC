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



def test_file_structure():

    for seq in SEQs:
        seq_dir = os.path.join(ROOT_DIR, seq)
        print(f"Testing file_structure on: {seq}")
        fs = file_structure(ROOT_DIR, seq, verbose=True)
        # Check target_dir
        assert fs._get_target_dir() == seq_dir
        # Check pose file loaded
        positions = fs._get_positions_()
        assert positions is not None and len(positions) > 0, "Pose file is empty or not loaded"

        # Plot path, colored with the label
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        aligned_positions = aligned_path(positions)

        # Linear elevation
        elevated_positions = elevate_along_path(aligned_positions, max_elevation=10.0)

        # Or smooth elevation
        # elevated_positions = elevate_along_path_smooth(aligned_positions, max_elevation=10.0, smoothness=1.0)

        # get label distribution
        label_distribution = fs._get_labels()
        print(f"Label distribution for {seq}: {np.unique(label_distribution, return_counts=True)}")
        
        # 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create color array from labels
        colors_rgb = [COLORS[label] for label in label_distribution]
        
        ax.scatter(elevated_positions[:, 0], elevated_positions[:, 1], elevated_positions[:, 2],
                   c=colors_rgb, s=10, alpha=0.6)
        ax.plot(elevated_positions[:, 0], elevated_positions[:, 1], elevated_positions[:, 2],
                'k-', alpha=0.3, linewidth=0.5)  # Connect points with a line
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z (Elevation)")
        ax.set_title(f"3D Path for sequence: {seq}")
        
        # Keep equal aspect ratio
        max_range = np.array([elevated_positions[:, 0].max()-elevated_positions[:, 0].min(),
                              elevated_positions[:, 1].max()-elevated_positions[:, 1].min(),
                              elevated_positions[:, 2].max()-elevated_positions[:, 2].min()]).max() / 2.0
        
        mid_x = (elevated_positions[:, 0].max()+elevated_positions[:, 0].min()) * 0.5
        mid_y = (elevated_positions[:, 1].max()+elevated_positions[:, 1].min()) * 0.5
        mid_z = (elevated_positions[:, 2].max()+elevated_positions[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()


if __name__ == '__main__':
    # Load test data
    print("*"*10)
    print("\n")
    
    test_file_structure()

    print("*"*10)
    print("\n")

    print("All tests passed!")
