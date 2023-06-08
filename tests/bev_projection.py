
import sys, os
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from dataloader.projections import BEVProjection
from PIL import Image
import numpy as np

def test_bev(file,width,height):
    data_handler = BEVProjection(width,height)
    input = data_handler.load(file)

    w,h = input.shape
    assert w == width,'Width is wrong'
    assert h == height,'Height is wrong'
    return True

if __name__ == "__main__":
    width = 512
    height = 512

    file = 'tutorial_data/000000.bin'

    flag = test_bev(file,width,height)
    print("BEV is passing")

    # Save Projection
    data_handler = BEVProjection(width,height)
    input = data_handler.load(file)

    proj_height = input['height']
    proj_height_np = proj_height.astype(np.uint8).squeeze()
    im_proj_pil = Image.fromarray(proj_height_np)
    im_proj_pil.save('proj_bev.png')
