
import os
import yaml

# Define the path to the dataset

test_sequences = [#'PCD_Easy_DARK',
                  #'PCD_MED',
                  #'PCD_EASY',
                  'PCD_RAS_EASY',
                  'PCD_RAS_MED',
                ]

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  # get parent dir 
print("Root directory:", root)
session_cfg_file = os.path.join(root,'PointNetGAP','sessions', 'hortov2.yaml')

output_cfg_file = os.path.join(root,'PointNetGAP','sessions', 'hortov2_output.yaml')
SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

# Save the SESSION dict to a file
def save_session_config(session_dict, filename='session_config.yaml'):
    """Save SESSION configuration dict to YAML file."""
    with open(filename, 'w') as f:
        yaml.dump(session_dict, f, default_flow_style=False)
    print(f"[INFO] Saved SESSION config to {filename}")

for seq in test_sequences:
        
        SESSION['val_loader']['dataset']['seq'] = [seq]
        
        for network in ['PointNetPGAP','PointNetVLAD','SPVSoAP3D', 'LOGG3D','overlap_transformer']:
                checkpoints = f"checkpoints/hortov2/{network}-LazyTripletLoss_L2/best_model.pth"    
                # checkpoints = ""
                path_checkpoint = os.path.exists(os.path.join(root,'PointNetGAP',checkpoints))
                assert os.path.exists(path_checkpoint), f"Checkpoint file not found {path_checkpoint}"
                
                SESSION['network']['checkpoints'] = checkpoints
                SESSION['network']['architecture'] = network
                
                # Save SESSION config for this run
                save_session_config(SESSION, output_cfg_file)
                
                os.system(f'python gen_descriptors.py --session hortov2_output')

                os.system(f'rm {output_cfg_file}')