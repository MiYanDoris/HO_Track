import numpy as np
import os

YCBModelsDir = '/mnt/data/hewang/h2o_data/HO3D/models/'
obj_lst = os.listdir(YCBModelsDir)

for obj in obj_lst:
    pcd = np.loadtxt(os.path.join(YCBModelsDir, obj, 'obj_2048.txt'))
    n_max = np.max(pcd, axis=0)
    n_min = np.min(pcd, axis=0)
    scale = np.linalg.norm(n_max - n_min)
    print(obj, scale)
    np.savetxt(os.path.join(YCBModelsDir, obj, 'scale.txt'), np.array([scale]))
