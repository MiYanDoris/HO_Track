import os
import numpy as np
import open3d as o3d
from datasets.ho3d.utils.vis_utils import *
from datasets.data_utils import farthest_point_sample
import torch

YCBModelsDir = '/mnt/data/hewang/h2o_data/HO3D/models/'
obj_lst = os.listdir(YCBModelsDir)

for obj in obj_lst:
    mesh = o3d.geometry.TriangleMesh()
    objMesh = read_obj(os.path.join(YCBModelsDir, obj, 'textured_simple.obj'))
    mesh.vertices = o3d.utility.Vector3dVector(np.copy(objMesh.v))
    obj_pcd = np.asarray(mesh.vertices)
    sample_idx = farthest_point_sample(obj_pcd, 2048, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    cam_points = obj_pcd[sample_idx]
    np.savetxt(os.path.join(YCBModelsDir, obj, 'obj_2048.txt'), cam_points)