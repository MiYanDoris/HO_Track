import numpy as np
import open3d as o3d
import cv2

height, width = 1080, 1920

def get_depth_from_pcld(pcld, K):
    depth = np.zeros((height, width))
    tmp = pcld / pcld[:, 2:]
    xy = (np.matmul(K, tmp.T).T[:, :2]) # N, 2
    xy = xy.astype(int)
    x = xy[:, 0]
    y = xy[:, 1]
    depth[y, x] = pcld[:, 2]
    return depth

def generate_clean_depth(K):
    pcd_pth = '/mnt/data/hewang/h2o_data/HOI4D/final_bottle/N03/refine_pc/0.pcd'
    pcd = o3d.io.read_point_cloud(pcd_pth)
    pcld = np.asarray(pcd.points)
    clean_depth = get_depth_from_pcld(pcld, K)
    clean_depth

K = np.load('/mnt/data/hewang/h2o_data/HOI4D/final_bottle/color_intrin.npy')
generate_clean_depth(K)