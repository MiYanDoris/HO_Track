
import numpy as np
import open3d as o3d
import cv2
import os
# from filter import read_mask_crop_wrist
from manopth.manolayer import ManoLayer
import pickle
import torch
import trimesh
import json
from scipy.spatial.transform import Rotation as Rt
from network.models.our_mano import OurManoLayer

mano_layer_right = ManoLayer(
            mano_root='/home/hewang/jiayi/manopth/mano/models' , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)

mano_layer_right_our = OurManoLayer()

data_dir = '/mnt/data/hewang/h2o_data/HOI4D/C5/N21/S56/s3/T1'

camera_id = 2
color_intrin = np.load('/mnt/data/hewang/h2o_data/HOI4D/camera_params/ZY2021080000%s/color_intrin.npy' % camera_id)
print(color_intrin)

width = 1920
height = 1080

xmap = np.array([[j for i in range(width)] for j in range(height)])
ymap = np.array([[i for i in range(width)] for j in range(height)])

def dpt_2_cld(dpt, K):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)

    if len(choose) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)  # position in camera reference

    return cld, choose

def outlier_removal(hand_cld, gt_vertices):
    distance = np.linalg.norm(hand_cld[None, :, :] - gt_vertices[:, None, :], axis=2).min(axis=0)
    fg_mask = np.where(distance < 0.05)[0] 
    hand_cld = hand_cld[fg_mask]
    return hand_cld

def get_full_pcld(data_dir, instance):
    depth_pth = os.path.join(data_dir, 'align_depth/%d.png' % instance)
    depth = o3d.io.read_image(depth_pth)

    mask_pth = os.path.join(data_dir, '2Dseg/palm_mask/%05d.png' % instance)
    mask = cv2.imread(mask_pth)
    label = np.ones((height, width))
    mask_sum = np.sum(mask, axis=-1)

    back_idx = np.where(mask_sum != 128)
    obj_idx = np.where(mask[:, :, 2] != 128)
    label[back_idx] = -1
    label[obj_idx] = 0
    label = label.reshape(width*height)

    depth_masked = np.asarray(depth)
    depth_masked[back_idx] = 0.0
    depth_masked = depth_masked / 1000.0

    cld, choose = dpt_2_cld(depth_masked, color_intrin)
    label = label[choose]

    return cld, label

def get_hand_anno(data_dir, instance, name):
    pkl_path = os.path.join(data_dir, 'hand_pose_refined/%d.pickle' % instance)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    mano_para = data['poseCoeff']
    mano_beta = data['beta']
    trans = data['trans']

    vertices, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(mano_para).unsqueeze(0),
                                                    th_trans=torch.zeros((1, 3)),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10)
                                                    )
    vertices = vertices[0]/1000
    vertices += trans
    hand_kp = hand_kp[0]/1000
    hand_kp  += trans
    print(hand_kp)

    vertices, hand_kp = mano_layer_right_our.forward(th_pose_coeffs=torch.FloatTensor(mano_para).unsqueeze(0),
                                                    th_trans=torch.FloatTensor(trans).reshape(1, 3),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10)
                                                    )
    print(hand_kp)
    exit(0)
    hand_faces = mano_layer_right.th_faces.cpu().data.numpy()

    mesh = trimesh.Trimesh(vertices=vertices, faces=hand_faces)
    mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False,
                                                include_texture=False, return_texture=False, write_texture=False,
                                                resolver=None, digits=8)
    with open(name, "w") as fp:
        fp.write(mesh_txt)

def get_bbox(data_dir, instance):
    obj_pth = os.path.join(data_dir, 'objpose_refined/%04d.json' % instance)
    f = open(obj_pth)
    obj_pose = json.load(f)
    obj_center_dict = obj_pose['dataList'][0]['center']
    obj_center = torch.zeros(3)
    obj_center[0] = obj_center_dict['x']
    obj_center[1] = obj_center_dict['y']
    obj_center[2] = obj_center_dict['z']

    bbox_length = torch.zeros(3)
    bbox_length[0] = obj_pose['dataList'][0]['dimensions']['height']/2
    bbox_length[1] = obj_pose['dataList'][0]['dimensions']['length']/2
    bbox_length[2] = obj_pose['dataList'][0]['dimensions']['width']/2

    corner = torch.zeros((8, 3))
    for i in range(8):
        weight_str = bin(i)[2:]
        weight_str = weight_str.zfill(3)
        weight = torch.tensor([int(x) for x in weight_str]) * 2 - 1
        corner[i] = weight * bbox_length

    rot = obj_pose['dataList'][0]['rotation']
    rot = np.array([rot['x'], rot['y'], rot['z']])
    rotation_mat = Rt.from_euler('XYZ', rot).as_matrix()

    bbox_rotated = np.matmul(corner, rotation_mat)
    bbox_rotated = obj_center + bbox_rotated
    np.savetxt('bbox_rotated.txt', bbox_rotated)
    np.savetxt('corner.txt', corner)
    np.savetxt('center.txt', obj_center.numpy().T[None, :])

    exit(0)

for i in range(1, 20):
    id = 12
    get_hand_anno(data_dir, id, './real_vis/hand_%d.obj' % id)
    # get_bbox(data_dir, id)
    # palm_mask_pth = os.path.join(data_dir, '2Dseg/palm_mask/%05d.png' % id)
    # if not os.path.exists(palm_mask_pth):
    #     continue
    # cld, label = get_full_pcld(data_dir, instance=id)
    # pcld = np.concatenate([cld, label[:, None]], axis=-1)
    # np.savetxt('./real_vis/test_%d.txt' % id, pcld)
    # get_hand_anno(data_dir, id, './real_vis/hand_%d.obj' % id)
