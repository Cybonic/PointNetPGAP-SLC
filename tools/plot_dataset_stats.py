import os,sys
import yaml
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from pipeline_factory import dataloader_handler

device_name = os.uname()[1]
pc_config = yaml.safe_load(open("sessions/pc_config.yaml", 'r'))
root_dir = pc_config[device_name]


dataset = "uk"

session_cfg_file = os.path.join('sessions',dataset + '.yaml')
print("Opening session config file: %s" % session_cfg_file)
SESSION = yaml.safe_load(open(session_cfg_file, 'r'))

seq_names_dict = SESSION['train_sequence']

for test,train in seq_names_dict.items():
    print("\n")
    print("="*30)
    print("test: %s" % test)
  
    SESSION['train_loader']['sequence'] = train
    SESSION['val_loader']['sequence'] = [test]
    
    loader = dataloader_handler(root_dir,"ResNet50_ORCHNet",dataset,SESSION)

    train_loader = loader.get_train_loader()
    val_loader = loader.get_val_loader()

    print("\n")
    print("*"*30)
    print(test)
    print("train_loader: %s" % len(train_loader))
    print("test_loader: %s" % len(val_loader))