"""Unit tests for the dataset loading and processing functions.
"""

# Add global path to sys path 
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from PointNetGAP.dataloader.hortov2.dataset import load_from_csv,file_structure


ROOT_DIR = os.path.abspath("/home/tiago/workspace/place_uk/dataset/PlaceRecognitionTestPolyTunnel")
SEQs = ["PCD_EASY",
        "PCD_Easy_DARK",
        "PCD_MED",
        "PCD_RAS_EASY"]

if __name__ == '__main__':
    # Load test data
    
    for seq in SEQs:
        print("*"*10)
        print(f"Testing sequence: {seq}")
        # Remove "pcd" from sequence name
        file_seq = seq.replace("PCD_", "")
        seq_dir = os.path.join(ROOT_DIR, seq)
        pose_csv_file = os.path.join(seq_dir,"path_{}".format(file_seq).lower() + ".csv")
        
        print(f"Loading test data from: {pose_csv_file}")
        test_data = load_from_csv(pose_csv_file)

        # Run tests
        assert not test_data.empty, "Test data is empty"
        assert 'timestamp' in test_data.columns, "Missing timestamp column"
        assert 'ID' in test_data.columns, "Missing ID column"

        
        file_structure()
        
        print("All tests passed!")
