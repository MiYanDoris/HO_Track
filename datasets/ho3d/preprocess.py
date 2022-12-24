import os
import numpy as np
import open3d as o3d
from os.path import join
import cv2
import argparse
import pickle
# from ho3d.utils.vis_utils import *
import sys
sys.path.append('..')
from data_utils import farthest_point_sample
import torch
import tqdm

# Paths and Params

height, width = 480, 640
depth_threshold = 800

multiCamSeqs = [
    'ABF1',
    'BB1',
    'GPMF1',
    'GSF1',
    'MDF1',
    'SB1',
    'ShSu1',
    'SiBF1',
    'SMu4',
    'MPM1',
    'AP1'
]


def inverse_relative(pose_1_to_2):
    pose_2_to_1 = np.zeros((4, 4), dtype='float32')
    pose_2_to_1[:3, :3] = np.transpose(pose_1_to_2[:3, :3])
    pose_2_to_1[:3, 3:4] = -np.dot(np.transpose(pose_1_to_2[:3, :3]), pose_1_to_2[:3, 3:4])
    pose_2_to_1[3, 3] = 1
    return pose_2_to_1


def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat

def read_depth_img(depth_filename, mode):
    """Read the depth image in dataset and decode it"""

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    mask = cv2.imread(depth_filename.replace(mode, 'mask').replace('depth', 'aligned_mask'))
    seg = np.zeros((height, width))
    hand_idx = (mask[:, :, 0] != 0)
    seg[hand_idx] = 1
    seg = seg.flatten()

    background = np.where(mask[:, :, 0] + mask[:, :, 1] == 0)
    dpt[background] = 0

    return dpt, seg

def get_obj(anno):
    mesh = o3d.geometry.TriangleMesh()
    YCBModelsDir = '/Users/yanmi/Desktop/hand-object-pose-tracking/data/ho3d/model/'
    objMesh = read_obj(os.path.join(YCBModelsDir, 'models', anno['objName'], 'textured_simple.obj'))
    objMesh.v = np.matmul(objMesh.v, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

    mesh.vertices = o3d.utility.Vector3dVector(np.copy(objMesh.v))
    mesh.triangles = o3d.utility.Vector3iVector(np.copy(objMesh.f))
    return mesh

xmap = np.array([[j for i in range(640)] for j in range(480)])
ymap = np.array([[i for i in range(640)] for j in range(480)])

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

def load_point_clouds(seq, fID, setDir, mode, K):
    seqDir = os.path.join(setDir, seq)

    path_depth = join(seqDir, 'depth', fID + '.png')

    depth_raw, mask = read_depth_img(path_depth, mode)
    if seq[-2].isnumeric():
        calibDir = os.path.join(setDir, '../', 'calibration', seq[:-1], 'calibration')
        K = get_intrinsics(join(calibDir, 'cam_{}_intrinsics.txt'.format(seq[-1]))).tolist()
    else:
        K = K
    cld, choose = dpt_2_cld(depth_raw, K)
    mask = mask[choose]
    cld[:, 1] *= -1
    cld[:, 2] *= -1
    return cld, mask

def outlier_removal(cld, mask, obj, hand_kp, sample_num=2048, device='cpu'):
    pcd = cld
    if len(pcd) > sample_num * 2:
        delta = len(pcd) // (sample_num * 2)
        pcd = pcd[::delta]
        mask = mask[::delta]

    # hand outlier removal
    hand_idx = (mask == 1)
    hand_cld = pcd[hand_idx]
    hand_label = mask[hand_idx]
    distance = np.linalg.norm(hand_cld[None, :, :] - hand_kp[:, None, :], axis=2).min(axis=0)
    fg_mask = np.where(distance < 0.05)[0]
    hand_cld = hand_cld[fg_mask]
    hand_label = hand_label[fg_mask]

    # obj outlier removal
    obj_idx = (mask == 0)
    obj_cld = pcd[obj_idx]
    obj_label = mask[obj_idx]
    distance = np.linalg.norm(obj_cld[None, :, :] - obj[:, None, :], axis=2).min(axis=0)
    fg_mask = np.where(distance < 0.01)[0]
    obj_cld = obj_cld[fg_mask]
    obj_label = obj_label[fg_mask]

    pcd = np.concatenate([hand_cld, obj_cld], axis=0)
    label = np.concatenate([hand_label, obj_label], axis=0)

    if len(pcd) > sample_num:
        sample_idx = farthest_point_sample(pcd, sample_num, device)
        label = label[sample_idx]
        pcd = pcd[sample_idx]

    return pcd, label

def get_anno(seq, fID, mode):
    f_name = '/data/h2o_data/HO3D/%s/%s/meta/%s.pkl' % (mode, seq, fID)
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def preprocess(args, seq, fID, smaple_num, device, mode):
    setDir = join(args.base_path, mode)

    anno = get_anno(seq, fID, mode)
    obj_name = anno['objName']
    obj_pc_pth = join('/data/h2o_data/HO3D/models/', obj_name, 'obj_2048.txt')
    obj_cld = np.loadtxt(obj_pc_pth)
    obj_cld = np.matmul(obj_cld, cv2.Rodrigues(anno['objRot'])[0].T) + anno['objTrans']

    if not seq[-2].isnumeric():
        K = anno['camMat']
    else:
        K = None
    # cld, mask = load_point_clouds(seq, fID, setDir, mode, K)
    # cld, mask = outlier_removal(cld, mask, obj_cld,anno['handJoints3D'], smaple_num, device=device)

    YCBModelsDir = '/data/h2o_data/HO3D/models/'
    scale_pth = os.path.join(YCBModelsDir, anno['objName'], 'scale.txt')
    scale = np.loadtxt(scale_pth)

    data = {
        # 'points': cld,
        # 'labels': mask,
        'obj_pose':{
            'translation': anno['objTrans'],
            'ID': seq[:-1],
            'rotation': cv2.Rodrigues(anno['objRot'])[0],
            'scale': scale,
            'CAD_ID': anno['objName'],
        },
        'hand_pose':{
            'pose': anno['handPose'],
            'translation': anno['handTrans'],
            'handJoints3D': anno['handJoints3D'],
            'beta': anno['handBeta'],
        },
        'file_name': '%s/%s' % (seq, fID)
    }
    data_dir = os.path.join(args.base_path, mode, seq, 'pcd_raw')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data_pth = os.path.join(data_dir, fID)
    np.save(data_pth, data, allow_pickle=True)
    return

# def add_scale(args, seq, cID, fID, smaple_num, device, mode):
#     data_dir = os.path.join(args.base_path, mode, '%s%s'%(seq, cID), 'pcd')
#     data_pth = os.path.join(data_dir, fID + '.npy')
#     print(data_pth)
#     cloud_dict = np.load(data_pth, allow_pickle=True).item()
#     anno = get_anno(seq, fID, cID, mode)

#     YCBModelsDir = '/data/h2o_data/HO3D/models/'
#     scale_pth = os.path.join(YCBModelsDir, anno['objName'], 'scale.txt')
#     scale = np.loadtxt(scale_pth)
#     cloud_dict['obj_pose']['scale'] = scale
#     cloud_dict['obj_pose']['CAD_ID'] = anno['objName']
#     cloud_dict['hand_pose']['beta'] = anno['handBeta']
#     np.save(data_pth, cloud_dict, allow_pickle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_num', type=int, default=2048,
                        required=False)
    parser.add_argument('--mode', type=str, default='train',
                        required=False)
    parser.add_argument('--ins', type=str, default='others')
    args = parser.parse_args()
    args.base_path = '/data/h2o_data/HO3D/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seq_lst = []
    fID_lst = []

    split_file_name = f'finalv2_test_{args.ins}.npy'
    print('preprocess ', split_file_name)
    split_file_pth = os.path.join('/data/h2o_data/HO3D/splits', split_file_name)
    test_data_dict = np.load(split_file_pth, allow_pickle=True).item()
    for seq in test_data_dict.keys():
        for segment in test_data_dict[seq].keys():
            idx_lst = test_data_dict[seq][segment]
            seq_lst.extend([seq] * len(idx_lst))
            fID_lst.extend(idx_lst)

    for seq, fID in tqdm.tqdm(zip(seq_lst, fID_lst), total=len(seq_lst)):
        file_pth = os.path.join(args.base_path, args.mode, seq, 'pcd_raw', '%04.npy')
        if not os.path.exists(file_pth):
            preprocess(args, seq, '%04d' % fID, args.sample_num, device, args.mode)
        # add_scale(args,  seq, cID, fID, args.sample_num, device, args.mode)
    # f_name = '/data/h2o_data/HO3D/train/SM2/meta/0000.pkl'
    # with open(f_name, 'rb') as f:
    #     try:
    #         pickle_data = pickle.load(f, encoding='latin1')
    #     except:
    #         pickle_data = pickle.load(f)

    # print(pickle_data.keys())
    # print(pickle_data['camMat'])