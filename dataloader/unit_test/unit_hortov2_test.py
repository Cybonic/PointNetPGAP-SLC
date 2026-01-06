"""Unit tests for the dataset loading and processing functions.
"""

# Add global path to sys path 
import sys
import os
from turtle import pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from PointNetGAP.dataloader.hortov2.dataset import load_from_csv,file_structure


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
    print("*"*10)
    print("\n")


def test_file_structure():

    for seq in SEQs:
        seq_dir = os.path.join(ROOT_DIR, seq)
        print(f"Testing file_structure on: {seq}")
        fs = file_structure(ROOT_DIR,seq)
        # Check target_dir
        assert fs._get_target_dir() == seq_dir
        # Check pose file loaded
        pose = fs._get_pose_()
        assert pose is not None and not pose.empty, "Pose file is empty or not loaded"
        # Check point cloud files
        pcl_files = fs._get_point_cloud_file_()
        assert isinstance(pcl_files, list) or isinstance(pcl_files, pd.Series)
        assert len(pcl_files) > 0, "No point cloud files found"

        
        
if __name__ == '__main__':
    # Load test data

    test_LOAD_FROM_CSV()

    test_file_structure()


    print("All tests passed!")
