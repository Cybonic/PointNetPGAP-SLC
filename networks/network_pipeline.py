import os
import sys
sys.path.append(os.path.dirname(__file__))
from pipelines.PointNetVLAD import *
from pipelines.LOGG3D import *
from pipelines.ORCHNet import *
from pipelines.overlap_transformer import *
from pipelines.PointNetVLAD import *

MODELS = ['LOGG3D','PointNetVLAD','ORCHNet','overlap_transformer']

def get_pipeline(pipeline_name, num_points=4096,output_dim=256):

    print("\n**************************************************")
    print(f"Model: {pipeline_name}")
    print(f"N.points: {num_points}")
    print(f"Dpts: {output_dim}")
    print("**************************************************\n")

    assert pipeline_name in MODELS
    if pipeline_name == 'LOGG3D':
        pipeline = LOGG3D(output_dim=output_dim)
    elif pipeline_name == 'PointNetVLAD':
        pipeline = PointNetVLAD( use_tnet=True, output_dim=output_dim, num_points=num_points,feat_dim = 1024)
    elif pipeline_name == 'ORCHNet':
        pipeline = ORCHNet('pointnet',output_dim=output_dim, num_points=num_points,feat_dim = 1024)
    elif pipeline_name == 'overlap_transformer':
        pipeline = featureExtracter(channels=1,height=64, width=900, output_dim=output_dim, use_transformer = True,
                                    feature_size=1024, max_samples=num_points)    
    return pipeline


if __name__ == '__main__':

    
    for model_name in MODELS:
        model = get_pipeline(model_name).cuda()

