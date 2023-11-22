
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.projections import SphericalProjection
from PIL import Image
import numpy as np
import yaml


def test_spherical(file,width,height,sensor= "HDL32"):

    sensor_pram = yaml.load(open("dataloader/sensor-cfg.yaml", 'r'),Loader=yaml.FullLoader)

    param = sensor_pram[sensor]
    fov_up = param["fov_up"]
    fov_down = param["fov_down"]
    width = param['SP']["width"]
    height = param['SP']["height"]
    
    data_handler = SphericalProjection(fov_up = fov_up,fov_down=fov_down,width=width,height=height) # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    input = data_handler.load(file)
    input = data_handler.to_tensor(input['range'])
    c,h,w = input.shape
    assert w == width,'Width is wrong'
    assert h == height,'Height is wrong'
    return True,input

def run_spherical_test(file,width,height):
    flag,proj_height = test_spherical(file,width,height)
    print("[PASSED] Spherical")
    return proj_height

if __name__ == "__main__":
    width = 1024
    height = 32

    file = '../data/orchards_aut22.bin'
    proj_height = run_spherical_test(file,width,height)

    # Save Projection
    proj_height_np = proj_height.numpy().astype(np.uint8).squeeze()
    im_proj_pil = Image.fromarray(proj_height_np)
    im_proj_pil.save('proj_spherical.png')
