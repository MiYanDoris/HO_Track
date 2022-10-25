import numpy as np
import os
import cv2
import pickle
import torch
from manopth.manolayer import ManoLayer
import tqdm

mano_layer_right = ManoLayer(
            mano_root='/home/hewang/jiayi/manopth/mano/models' , side='right', use_pca=False, ncomps=45, flat_hand_mean=True).cuda()

camera_id = 2
color_intrin = np.load('/mnt/data/hewang/h2o_data/HOI4D/camera_params/ZY2021080000%s/color_intrin.npy' % camera_id)

def get_color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap

def get_wrist_vertline(wrist, mean):
    #y=kx+b, k = (mean[1] - wrist[1])/(mean[0] - wrist[0]), b = wrist[1]-k*wrist[0]
    #b' = wrist[1]
    # k_vert = -(mean[0] - wrist[0])/(mean[1] - wrist[1])
    # b_vert = wrist[1]-k_vert*wrist[0]
    k = (mean[1] - wrist[1])/max(0.001, (mean[0] - wrist[0]))
    b = wrist[1]-k*wrist[0]

    bias = 40
    bias_x = wrist[0]
    bias_y = k*bias_x+b+bias

    k_vert = -(mean[0] - wrist[0])/(mean[1] - wrist[1])
    b_vert = bias_y-k_vert*bias_x

    return k_vert,  b_vert

def read_raw_seg(maskfilepth, h=1920, w=1080):  # right 2 left 3
    color_map = get_color_map()
    mask = cv2.imread(maskfilepth)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (h, w))
    new_mask = np.zeros(mask.shape)
    hand_idx_left = np.where((mask == color_map[3]).all(axis=2))#H2O的时候要反过来
    hand_idx_right = np.where((mask == color_map[2]).all(axis=2))
    obj_idx = np.where((mask == color_map[1]).all(axis=2))

    new_mask[hand_idx_left] = [3, 3, 3]
    new_mask[hand_idx_right] = [2, 2, 2]
    new_mask[obj_idx] = [1, 1, 1]
    mask[hand_idx_right] = 255
    return new_mask

def read_mask_crop_wrist(file,kps_anno,docrop = True, hand_color=2, obj_color=1, dscale=1, crop=(0, 1080, 0, 1920),h=1920,w=1080):
    h, w = int(round((crop[3]-crop[2]) / dscale)
               ), int(round((crop[1]-crop[0]) / dscale))
    mask = read_raw_seg(file, h, w)
    mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]
    # print('seg!',np.unique(mask))

    hand_idx = np.where((mask == hand_color).all(axis=2))
    obj_idx = np.where((mask == obj_color).all(axis=2))
    hand_mask = np.zeros((w, h, 3), dtype=np.float32)
    obj_mask = np.ones((w, h, 3), dtype=np.float32)
    obj_mask[obj_idx] = [0, 0, 0]
    hand_mask[hand_idx] = [0, 128, 0]  # TODO: double hands
    hand_mask[obj_idx] = [0, 0, 128]
    if docrop:
        mean = np.mean(kps_anno, axis=0)
        for kps_coord in kps_anno:
            cv2.circle(mask, center=(int(kps_coord[0]), int(
                kps_coord[1])), radius=3, color=[255, 255, 0], thickness=-1)
            cv2.circle(mask, center=(int(mean[0]), int(mean[1])), radius=3, color=[
                    255, 0, 255], thickness=-1)
        k, b = get_wrist_vertline(kps_anno[0], mean)
        print(k, b)

        for index_cnt in range(len(hand_idx[1])):
            if(k*hand_idx[1][index_cnt]+b < hand_idx[0][index_cnt]):
                mask[hand_idx[0][index_cnt], hand_idx[1][index_cnt]] = [255, 255, 255]
                hand_mask[hand_idx[0][index_cnt],hand_idx[1][index_cnt]] = [0, 0, 0]

    return hand_mask

def get_hand_anno(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    mano_para = data['poseCoeff']
    mano_beta = data['beta']
    trans = data['trans']

    _, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(mano_para).unsqueeze(0).cuda(),
                                                    th_trans=torch.zeros((1, 3)).cuda(),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10).cuda()
                                                    )
    hand_kp = (hand_kp[0]/1000).cpu().numpy()
    hand_kp += trans
    return hand_kp

def world22D(hand_kp, K):
    tmp = hand_kp / hand_kp[:, 2:]
    xy = np.matmul(K, tmp.T).T[:, :2]
    return xy

# root_dir = '/mnt/data/hewang/h2o_data/HOI4D/C5'
# task_list = []
# txt_pth = '/mnt/data/hewang/h2o_data/HOI4D/C5/bottle_refined_pose.txt'
# with open(txt_pth, "r", errors='replace') as fp:
#     lines = fp.readlines()
#     for i in lines:
#         _, obj_id, scene, layout, task = i.strip().split('/')
#         task_list.append(os.path.join(root_dir, obj_id, scene, layout, task))

root_dir = '/mnt/data/hewang/h2o_data/HOI4D/final/car'
# task_list = ['N06', 'N18', 'N21', 'N23', 'N28']
# task_list = ['N06', 'N23']
task_list = ['N22', 'N24', 'N25', 'N34']

task_dir = [os.path.join(root_dir, task) for task in task_list]

# for task in tqdm.tqdm(task_dir):
#     for instance in tqdm.tqdm(range(300)):
#         pkl_path = os.path.join(task, 'hand_pose_refined/%d.pickle' % instance)
#         if os.path.exists(pkl_path):
#             hand_kp = get_hand_anno(pkl_path)
#             xy = world22D(hand_kp, color_intrin)
#             mask_pth = os.path.join(task, '2Dseg/mask/%05d.png' % instance)
#             palm_mask = read_mask_crop_wrist(mask_pth, xy)
#             palm_mask_pth = os.path.join(task, '2Dseg/palm_mask/%05d.png' % instance)
#             dir = os.path.dirname(palm_mask_pth)
#             if not os.path.exists(dir):
#                 os.mkdir(dir)
            # cv2.imwrite(palm_mask_pth, palm_mask)
for task in tqdm.tqdm(task_dir):
    for instance in tqdm.tqdm(range(300)):
        hand_pose_file_id = (instance // 6) * 6
        pkl_path = os.path.join(task, 'handpose/%d.pickle' % hand_pose_file_id)
        hand_kp = get_hand_anno(pkl_path)
        xy = world22D(hand_kp, color_intrin)
        mask_pth = os.path.join(task, '2Dseg/mask/%05d.png' % instance)
        palm_mask = read_mask_crop_wrist(mask_pth, xy)
        palm_mask_pth = os.path.join(task, '2Dseg/palm_mask/%05d.png' % instance)
        dir = os.path.dirname(palm_mask_pth)
        if not os.path.exists(dir):
            os.mkdir(dir)