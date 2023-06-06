
import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.projections import SphericalProjection
from PIL import Image
import numpy as np

def test_spherical(file,width,height):
    data_handler = SphericalProjection(height=height,width=width) # fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50
    input = data_handler.load(file)
    input = data_handler.to_tensor(input['range'])
    c,h,w = input.shape
    assert w == width,'Width is wrong'
    assert h == height,'Height is wrong'
    return True,input

width = 900
height = 64

file = 'tutorial_data/000000.bin'
flag,proj_height = test_spherical(file,width,height)
print("Spherical is passing")

# Save Projection
proj_height_np = proj_height.numpy().astype(np.uint8).squeeze()
im_proj_pil = Image.fromarray(proj_height_np)
im_proj_pil.save('proj_spherical.png')
