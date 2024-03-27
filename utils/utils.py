from datetime import datetime
from tqdm import tqdm
import torch
import numpy as np 

import GPUtil


def dump_info(file, text, flag='w'):
    
    now = datetime.now()
    current_time = now.strftime("%d|%H:%M:%S")
    
    f = open('results/' + file,flag)
    
    line = "{}||".format(now)

    if isinstance(text,dict):
        for key, values in text.items():
            line += "{}:{} ".format(key,values)
            
    elif isinstance(text,str):
        line += text
        #f.write(line)
    f.write(line + '\n')
    f.close()
    return(line)




def generate_descriptors(model,val_loader, device):
    model.eval()
    
    dataloader = iter(val_loader)
    tbar = tqdm(range(len(val_loader)), ncols=100)

    prediction_bag = {}
    idx_bag = []
    for batch_idx in tbar:
        input,inx = next(dataloader)
        if device in ['gpu','cuda']:
            input = input.to(device)
            input = input.cuda(non_blocking=True)
        
        if len(input.shape)<4:
            input = input.unsqueeze(0)
            
        if not torch.is_tensor(inx):
            inx = torch.tensor(inx)
        # Generate the Descriptor
        prediction = model(input)
        # Keep descriptors
        for d,i in zip(prediction.detach().cpu().numpy().tolist(),inx.detach().cpu().numpy().tolist()):
            prediction_bag[i] = d
    return(prediction_bag)




def unique2D(input):
    if not isinstance(input,(np.ndarray, np.generic)):
        input = np.array(input)
   
    
    output = []
    for p in input:
        output.extend(np.unique(p))
    #p = np.array([np.unique(p) for p in positive]).ravel()
    output = np.unique(output)
    return(output)




def get_available_devices(n_gpu,logger = None):
        if logger is None:
            import logging
            logger = logging.getLogger('utils')
        sys_gpu = torch.cuda.device_count()
        device = 'cpu'
        if sys_gpu == 0:
            logger.warning('No GPUs detected, using the CPU')
            n_gpu = 0
        elif n_gpu <= sys_gpu:
            logger.warning(f'Nbr of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu            
            gpu = GPUtil.getAvailable(order = 'first', limit = 1, maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])
            if len(gpu) > 0:
                device_name = torch.cuda.get_device_name()
                
                GPUtil.showUtilization()
                device = torch.device(f'cuda:{gpu[0]}' if n_gpu > 0 and len(gpu) >0 else 'cpu')
            else:
                print('No GPU available')
                device_name = "cpu"
                
            print(f'GPU to be used: {device_name}\n')
        
        logger.info(f'Detected GPUs: {sys_gpu} Requested: {n_gpu}')
        available_gpus = list(range(n_gpu))
        return device, available_gpus