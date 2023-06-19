import os,sys
sys.path.append(os.sep.join(os.path.dirname(__file__).split(os.sep)[:-1]))
from tests.laserscan import run_laserscan_test
from tests.kitti_triplet import run_kitt_triplet_test
from tests.kitti_eval import run_kitt_eval_test
from tests.sparselaserscan import run_sparse_test
from tests.spherical_projection import run_spherical_test
from tests.overlap_transformer import run_transformer_test
from tests.LOGG3D import run_LOGG3D_test
from tests.pointnetvlad import run_PointNetVLAD_test
from tests.orchnet import run_orchnet_test

width = 900
height = 64

voxel_size=0.05
file = 'tutorial_data/000000.bin'

root_devices = {'deep-MS-7C37':'/media/deep/datasets/datasets',
                'tiago-deep':'/home/tiago/Dropbox/research/datasets'}

device_name = os.uname()[1]
root = root_devices[device_name]

run_kitt_eval_test(root)
run_kitt_triplet_test(root)
run_laserscan_test(file)
run_sparse_test(file,voxel_size)
run_spherical_test(file,width,height)
run_transformer_test(file,width,height)
run_orchnet_test(file)
run_LOGG3D_test(file,voxel_size)
run_PointNetVLAD_test(file)