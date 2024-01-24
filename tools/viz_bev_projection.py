
import sys, os
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.projections import BEVProjection
from PIL import Image
import numpy as np

def test_bev(file,width,height):

    square_roi = [{'xmin':-25,'xmax':0,'ymin':-25,'ymax':25,'zmin':-2,'zmax':1.5}]
    data_handler = BEVProjection(width,height,square_roi=square_roi)
    input = data_handler.load(file)

    height_mod  = input['height']
    density_mod = input['density']
    intensity_mod = input['intensity']

    w,h,b = height_mod.shape
    assert w == width,'Width is wrong'
    assert h == height,'Height is wrong'
    return True,input

if __name__ == "__main__":
    width = 256
    height = 256

    file = 'tutorial_data/000000.bin'
    file = '../data/orchards_sum22.bin'
    flag,input = test_bev(file,width,height)
    print("BEV is passing")

    proj_height = input['height']
    proj_height_np = proj_height.astype(np.uint8).squeeze()

    proj_height_np_color_inv = proj_height_np
    im_proj_pil = Image.fromarray(proj_height_np_color_inv)
    im_proj_pil.save('proj_bev.png')
