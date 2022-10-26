import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import torch
from copy import deepcopy
from tqdm import tqdm
from network.models.our_mano import OurManoLayer
from network.models.hand_utils import handkp2palmkp
from data_utils import farthest_point_sample, mat_from_rvec, split_dataset, jitter_hand_mano, jitter_obj_pose
from scipy.spatial.transform import Rotation as Rt
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from configs.config import get_config
from data_utils import farthest_point_sample, mat_from_rvec, split_dataset, jitter_hand_mano, jitter_obj_pose, pose_list_to_dict
import cv2
import open3d as o3d
import pickle

def get_hand_anno(data_dir, instance, mano_layer_right_our):
    pkl_path = os.path.join(data_dir, 'hand_pose_refined/%d.pickle' % instance)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    mano_para = data['poseCoeff']
    mano_beta = data['beta']
    trans = data['trans']

    vertices, hand_kp = mano_layer_right_our.forward(th_pose_coeffs=torch.FloatTensor(mano_para).unsqueeze(0).cuda(),
                                                    th_trans=torch.zeros((1, 3)).cuda(),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10).cuda(),
                                                    original_version=True)
    vertices = (vertices[0]/1000).cpu()
    hand_kp = (hand_kp[0]/1000).cpu()
    vertices += trans
    hand_kp += trans

    _, template_kp = mano_layer_right_our.forward(th_pose_coeffs=torch.zeros((1, 48)).cuda(),
                                                    th_trans=torch.zeros((1, 3)).cuda(),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10).cuda(),
                                                    original_version=True)
    template_kp = template_kp.cpu()
    template_trans = (template_kp[0] / 1000)[0].numpy()
    global_trans = trans + template_trans

    palm_template = handkp2palmkp(template_kp)[0] / 1000
    palm_template -= template_trans

    return vertices.numpy(), mano_para, mano_beta, trans, hand_kp.numpy(), global_trans, palm_template.numpy()

def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
    anno = eval(cont)["dataList"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    rot_matrix = Rt.from_rotvec(rot).as_matrix()

    return rot_matrix, trans, dim

def get_depth_from_pcld(pcld, K, height, width):
    depth = np.zeros((height, width))
    tmp = pcld / pcld[:, 2:]
    xy = (np.matmul(K, tmp.T).T[:, :2]) # N, 2
    xy = xy.astype(int)
    x = xy[:, 0]
    y = xy[:, 1]
    depth[y, x] = pcld[:, 2]
    return depth

def generate_clean_depth(K, pcd_pth):
    pcd = o3d.io.read_point_cloud(pcd_pth)
    pcld = np.asarray(pcd.points)
    clean_depth = get_depth_from_pcld(pcld, K, 1080, 1920)
    return clean_depth

def load_point_clouds_HOI4D(mask_path, palm_mask_path, cam_in_path, pcd_pth):
    camMat = np.load(cam_in_path)
    depth_raw = generate_clean_depth(camMat, pcd_pth)

    mask = (cv2.imread(mask_path)[360:]) / 70
    mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)[:, :, 0]

    obj_mask = (mask == 0)
    hand_mask = (mask == 1)

    palm_mask = cv2.imread(palm_mask_path)[:, :, 1] != 0
    hand_mask &= palm_mask

    obj_depth = np.array(depth_raw * obj_mask).astype(np.float32)
    hand_depth = np.array(depth_raw * hand_mask).astype(np.float32)

    depth3d_obj = o3d.geometry.Image(obj_depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])
    obj_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d_obj, intrinsics, stride=2)
    obj_pcd, _ = obj_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    obj_pcd = np.asarray(obj_pcd.points)
    center = np.mean(obj_pcd, axis=0)
    dis = np.linalg.norm(obj_pcd - center.reshape(1, 3), axis=-1)
    obj_pcd = obj_pcd[dis < 0.12]

    depth3d_hand = o3d.geometry.Image(hand_depth)
    hand_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d_hand, intrinsics, stride=2)
    # hand_pcd, _ = hand_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    hand_pcd = np.asarray(hand_pcd.points)
    center = np.mean(hand_pcd, axis=0)
    dis = np.linalg.norm(hand_pcd - center.reshape(1, 3), axis=-1)
    hand_pcd = hand_pcd[dis < 0.1]
    return obj_pcd, hand_pcd, camMat

def generate_HOI4D_data(seq, id, category, points_source, input_only, device, mano_layer_right_our, pred_obj_pose_dir, load_pred_obj_pose, num_points=1024, start_frame=0):
    cam_in_path = '/mnt/data/hewang/h2o_data/HOI4D/final/color_intrin.npy'
    pcd_pth = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/refine_pc/%d.pcd' % (category, seq, id)
    mask_path = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/pred_mask/%06d.png' % (category, seq, id)
    palm_mask_path = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/2Dseg/palm_mask/%05d.png' % (category, seq, id)
    anno_path = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/objpose_refined/%04d.json' % (category, seq, id)
    if not os.path.exists(anno_path):
        anno_path = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/objpose/%05d.json' % (category, seq, id)
        if not os.path.exists(anno_path):
            anno_path = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s/objpose/%d.json' % (category, seq, id)

    rot, trans, _ = read_rtd(anno_path)
    obj_pcld, hand_pcld, camMat = load_point_clouds_HOI4D(mask_path, palm_mask_path, cam_in_path, pcd_pth)
    
    # sampling
    sample_idx = farthest_point_sample(hand_pcld, num_points, device)
    hand_pcld = hand_pcld[sample_idx]

    sample_idx = farthest_point_sample(obj_pcld, num_points, device)
    obj_pcld = obj_pcld[sample_idx]
    
    # hand pose
    task_dir = '/mnt/data/hewang/h2o_data/HOI4D/final/%s/%s' % (category, seq)
    _, mano_para, mano_beta, mano_trans, hand_kp, global_trans, palm_template = get_hand_anno(task_dir, id, mano_layer_right_our)

    rotvec = mano_para[:3]
    hand_global_rotation = mat_from_rvec(rotvec)
    
    full_data = {
        'points': hand_pcld,
        'category': category,
        'file_name':'%s/%04d' % (seq, id),
        'gt_obj_pose':{
            'translation': trans.reshape(1, 3, 1),
            'rotation': rot.reshape(1, 3, 3),
            'up_and_down_sym': False
        },# don't need for hand tracking
        'gt_hand_pose':{
            'mano_trans': np.array(mano_trans).reshape(3), # 3
            'scale': 0.2,  
            'rotation': np.array(hand_global_rotation).reshape(1, 3, 3),
            'mano_pose':mano_para,
            'translation': global_trans,
            'mano_beta': mano_beta,
            'palm_template': np.array(palm_template)
        },
        'jittered_hand_kp': hand_kp,
        'gt_hand_kp': hand_kp,
        'jittered_obj_pose': {  
            'translation': trans.reshape(1, 3, 1),
            'rotation': rot.reshape(1, 3, 3)
        },# don't need for hand tracking
        'projection': {
            'cx': camMat[0, 2],
            'cy': camMat[1, 2],
            'fx': camMat[0, 0],
            'fy': camMat[1, 1],
            'h': 1080,
            'w': 1920,
        }# don't need for hand tracking
    }

    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s_0000.pkl' % (seq))
        pred_dict = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = id - start_frame
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose

    return full_data

class HOI4DDataset:
    def __init__(self, cfg, mode, kind):
        '''
        kind: 'single_frame' or 'seq'
        mode: use 'test' to replace 'val'
        '''
        self.cfg = cfg
        self.root_dset = cfg['data_cfg']['basepath']
        self.category = cfg['obj_category']
        # TODO testing dataset!
        if mode == 'val':
            mode = 'test'
        self.mode = mode
        self.kind = kind
        self.points_source = cfg['points_source']
        self.device = cfg['device']
        self.seq_lst = []
        self.fID_lst = []
        self.seq_start = []
        self.num_points = cfg['num_points']
        self.input_only = self.cfg['input_only']
        self.mano_layer_right_our = OurManoLayer(mano_root=cfg['mano_root'], side='right').cuda()
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None

        if self.category == 'bottle':
            task_list = ['N06', 'N18', 'N21', 'N23', 'N28']
        elif self.category == 'car':
            task_list = ['N22', 'N24', 'N25', 'N34']

        for task in task_list:
            self.seq_start.append(len(self.fID_lst))
            self.seq_lst.extend([task] * 300)
            self.fID_lst.extend(np.arange(300))
        self.seq_start.append(len(self.fID_lst))

        self.len = len(self.seq_lst)
        print('HOI4D mode %s %s: %d frames' % (self.mode, self.kind, self.len))

    def __getitem__(self, index):
        seq = self.seq_lst[index]
        fID = self.fID_lst[index]

        full_data = generate_HOI4D_data(seq, fID, self.category, self.points_source, self.input_only, self.device, self.mano_layer_right_our, self.pred_obj_pose_dir, self.load_pred_obj_pose, self.num_points)
        return full_data

    def __len__(self):
        return self.len


def visualize_data(data_dict):
    from vis_utils import plot3d_pts

    mano_pose = data_dict['gt_hand_pose']['mano_pose']
    trans = data_dict['gt_hand_pose']['translation']
    beta = data_dict['gt_hand_pose']['mano_beta']
    mano_layer_right = OurManoLayer()
    hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                                                th_trans=torch.FloatTensor(trans).reshape(1, -1), th_betas=torch.from_numpy(beta).unsqueeze(0))
    hand_vertices = hand_vertices.cpu().data.numpy()[0]
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    hand_input = np.matmul(data_dict['pred_seg_hand_points'] - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    obj_input = np.matmul(data_dict['pred_seg_obj_points'] - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices], [hand_input], [obj_input], [hand_input, obj_input]],
               show_fig=False, save_fig=True,
               save_folder='./HOI4D/',
               save_name=data_dict['file_name'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3.7_final_obj_bottle_test_HOI4D.yml', help='path to config.yml')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--kind', type=str, default='seq', choices=['single_frame', 'seq'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args, save=False)
    dataset = HOI4DDataset(cfg, args.mode, args.kind)
    print(dataset.len)
    for i in range(30):
        visualize_data(dataset[i * 50])
    # beta_lst = []
    # for i, data in enumerate(dataloader):
    #     print(data['gt_hand_pose']['beta'])
    # print(len(dataset))
    # for i in range(len(dataset)):
    #     visualize_data(dataset[i], cfg['obj_category'][0], cfg['mano_root'])
    #     if i == 150:
    #         break
       # hand_pc = dataset[i]['hand_kp']
       # np.save('/home/vipuser/Desktop/jiayi/hand-object-pose-tracking-temp/test_hand_kp/%05d.npy' % i, hand_pc)
    # file_lst = os.listdir('/mnt/data/hewang/h2o_data/HO3D/train')
    # file_lst.sort()
    # beta_lst = []
    # for seq in file_lst:
    #     pth = '/mnt/data/hewang/h2o_data/HO3D/train/%s/pcd/0000.npy' % seq
    #     if os.path.exists(pth):
    #         cloud_dict = np.load(pth, allow_pickle=True).item()
    #         beta = cloud_dict['hand_pose']['beta']
    #         if len(beta_lst) == 0 or np.sum(np.abs(beta - beta_lst[-1])) > 1e-7:
    #             beta_lst.append(cloud_dict['hand_pose']['beta'])
    # print(beta_lst)
    # print(len(beta_lst))
    # beta_array = np.array(beta_lst)
    # np.savetxt('beta.txt', beta_array)
