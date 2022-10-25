import os
import sys
import numpy as np
from os.path import join as pjoin
import torch
from network.models.our_mano import OurManoLayer
from network.models.hand_utils import handkp2palmkp
import open3d as o3d
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from data_utils import farthest_point_sample, mat_from_rvec, jitter_hand_kp, jitter_obj_pose, pose_list_to_dict
import cv2

height, width = 480, 640

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)
    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    mask = cv2.imread(depth_filename.replace('train', 'mask').replace('depth', 'aligned_mask'))
    seg = np.zeros((height, width))
    hand_idx = (mask[:, :, 0] != 0)
    seg[hand_idx] = 1

    background = np.where(mask[:, :, 0] + mask[:, :, 1] == 0)
    seg[background] = 2

    return dpt, seg

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

def load_point_clouds(root_dir, seq, fID, K):
    path_depth = os.path.join(root_dir, 'train/%s/depth/%s.png' % (seq, fID))
    depth_raw, _ = read_depth_img(path_depth)
    if seq[-2].isnumeric():
        calibDir = os.path.join(root_dir, 'calibration', seq[:-1], 'calibration')
        K = get_intrinsics(os.path.join(calibDir, 'cam_{}_intrinsics.txt'.format(seq[-1]))).tolist()
    else:
        K = K

    ensemble_mask_pth = os.path.join(root_dir, 'pred_mask/ensemble/%s/%s.png' % (seq, fID))
    mask = (cv2.imread(ensemble_mask_pth) / 70)[:240] # 0:obj,1:hand,2back use gt
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)[:, :, 0]
    mask = mask.flatten()

    cld, choose = dpt_2_cld(depth_raw, K)
    cld[:, 1] *= -1
    cld[:, 2] *= -1
    mask = mask[choose]
    
    hand_idx = (mask == 1)
    obj_idx = (mask == 0)
    hand_pcld = cld[hand_idx]
    obj_pcld = cld[obj_idx]
    del mask

    return hand_pcld, obj_pcld, K

def get_anno(root_dir, seq, fID):
    import pickle
    f_name = pjoin(root_dir, 'train/%s/meta/%s.pkl' % (seq, fID))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def generate_HO3D_data(mano_layer_right, root_dir, path, num_points, obj_perturb_cfg, hand_jitter_config, device, load_pred_obj_pose, pred_obj_pose_dir, start_frame, cur_frame, points_source=None):
    tmp = np.load(path, allow_pickle=True)
    cloud_dict = tmp.item()

    # get point cloud
    seq, fID = path.split('/')[-3], path.split('/')[-1][:4]
    if not seq[-2].isnumeric():
        anno = get_anno(root_dir, seq, fID)
        K = anno['camMat']
    else:
        K = None

    hand_pcld, obj_pcld, cam_Mat = load_point_clouds(root_dir, seq, fID, K)
    cam_cx, cam_cy = cam_Mat[0][2], cam_Mat[1][2]
    cam_fx, cam_fy = cam_Mat[0][0], cam_Mat[1][1]

    # generate obj canonical point clouds
    origin_obj_pose = cloud_dict['obj_pose']
    obj_pose = {}
    # obj_nocs = np.zeros_like(cam_points)

    # obj_idx = np.where(seg == 0)[0]
    # if len(obj_idx)!=0:
    #     obj_nocs[obj_idx] = np.matmul((cam_points[obj_idx] - np.expand_dims(origin_obj_pose['translation'], 0)) / origin_obj_pose['scale'],
    #                                 origin_obj_pose['rotation'])
    obj_pose['translation'] = np.expand_dims(np.array(origin_obj_pose['translation']), axis=1)
    obj_pose['rotation'] = origin_obj_pose['rotation']
    obj_pose['scale'] = origin_obj_pose['scale']


    #generate hand canonical point clouds
    mano_pose = np.array(cloud_dict['hand_pose']['pose'])
    hand_global_rotation = mat_from_rvec(mano_pose[:3])
    mano_trans = np.array(cloud_dict['hand_pose']['translation'])
    #hand keypoints gt
    reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    hand_kp = cloud_dict['hand_pose']['handJoints3D']
    hand_kp = hand_kp[reorder]
    world_trans = hand_kp[0]

    # remove outliers
    obj_dis = np.linalg.norm(obj_pcld - obj_pose['translation'].transpose(-1,-2), axis=-1)
    foreground = np.where(obj_dis < 0.25)
    obj_pcld = obj_pcld[foreground]
    hand_dis = np.linalg.norm(hand_pcld - hand_kp[9], axis=-1)
    foreground = np.where(hand_dis < 0.15)
    hand_pcld = hand_pcld[foreground]
    if hand_pcld.shape[0] == 0 or obj_pcld.shape[0] == 0:
        print(seq, fID)
    # sampling
    sample_idx = farthest_point_sample(hand_pcld, num_points, device)
    hand_pcld = hand_pcld[sample_idx]

    sample_idx = farthest_point_sample(obj_pcld, num_points, device)
    obj_pcld = obj_pcld[sample_idx]

    #shuffle
    n = obj_pcld.shape[0]
    perm = np.random.permutation(n)
    obj_pcld = obj_pcld[perm]

    n = hand_pcld.shape[0]
    perm = np.random.permutation(n)
    hand_pcld = hand_pcld[perm]
    
    #jitter hand pose and obj pose
    
    # if hand_jitter_config['jitter_by_MANO']:
    #     jittered_mano, jittered_trans, jittered_beta = jitter_hand_mano(torch.FloatTensor(hand_global_rotation), np.array(mano_pose[3:]), mano_trans, mano_beta, hand_jitter_config)
    #     _, jittered_hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(jittered_mano.reshape(1, -1)).cuda(),
    #                                             th_trans=torch.FloatTensor(jittered_trans).reshape(1, -1).cuda(),
    #                                             th_betas=torch.FloatTensor(jittered_beta).reshape(1, -1).cuda(), original_version=True)
    #     jittered_hand_kp = jittered_hand_kp[0].cpu().numpy()
    # else:
    #     jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)
    jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)

    rest_pose = torch.zeros((1, 48))
    rest_pose[0, 3:] = torch.FloatTensor(mano_pose[3:])
    _, template_kp = mano_layer_right.forward(th_pose_coeffs=rest_pose, th_trans=torch.zeros((1, 3)), th_betas=torch.FloatTensor(cloud_dict['hand_pose']['beta']).reshape(1, 10))
    palm_template = handkp2palmkp(template_kp)
    palm_template = palm_template[0].cpu().float()

    pose_perturb_cfg = {'type': obj_perturb_cfg['type'],
                        'scale': obj_perturb_cfg['s'],
                        'translation': obj_perturb_cfg['t'],  # Now pass the sigma of the norm
                        'rotation': np.deg2rad(obj_perturb_cfg['r'])}
    jittered_obj_pose_lst = []
    jittered_obj_pose = jitter_obj_pose(obj_pose, pose_perturb_cfg)
    jittered_obj_pose_lst.append(jittered_obj_pose)
    
    full_data = {
        'pred_seg_hand_points': hand_pcld,
        'pred_seg_obj_points': obj_pcld,
        'jittered_obj_pose': pose_list_to_dict(jittered_obj_pose_lst),
        'gt_obj_pose': pose_list_to_dict([obj_pose]),
        'jittered_hand_kp': jittered_hand_kp,
        'gt_hand_kp': hand_kp,
        'gt_hand_pose':{
                        'translation':world_trans,
                        'scale': 0.2,
                        'rotation': np.array(hand_global_rotation),
                        'mano_pose':mano_pose,
                        'mano_trans':mano_trans,
                        'mano_beta': cloud_dict['hand_pose']['beta'],
                        'palm_template': palm_template
                        },
        'category': cloud_dict['obj_pose']['CAD_ID'],
        'file_name':cloud_dict['file_name'],
        'projection': {'w':640, 'h':480, 'fx':-cam_fx,'fy':cam_fy,'cx':cam_cx,'cy':cam_cy},
    }

    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s_%04d.pkl' % (seq.replace('/', '_'), start_frame))
        tmp = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_dict = tmp
        del tmp
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = cur_frame - start_frame
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose

    if points_source == 'pred_seg_obj':
        full_data['points'] = obj_pcld
        full_data['labels'] = np.zeros_like(obj_pcld[:,0])
        full_data['other_points'] = hand_pcld
    elif points_source == 'pred_seg_hand':  
        full_data['points'] = hand_pcld
        full_data['other_points'] = obj_pcld
    elif points_source == 'all':        # for captra
        full_data['points'] = np.concatenate([obj_pcld, hand_pcld], axis=0)
        full_data['labels'] = np.concatenate([np.zeros(len(obj_pcld)), np.ones(len(hand_pcld))], axis=0)
        full_data['obj_nocs'] = np.zeros_like(full_data['points'])
    else:
        raise NotImplementedError

    if 'can' in full_data['category'] or 'box' in full_data['category']:
        full_data['gt_obj_pose']['up_and_down_sym'] = True
    else:
        full_data['gt_obj_pose']['up_and_down_sym'] = False

    return full_data

class HO3DDataset:
    def __init__(self, cfg, mode, kind):
        '''
            kind: 'single_frame' or 'seq'
            mode: use 'test' to replace 'val'
        '''
        print('HO3DDataset!')
        self.cfg = cfg
        self.root_dset = cfg['data_cfg']['basepath']
        self.category_lst = cfg['obj_category']
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']
        self.points_source = cfg['points_source']
        self.handframe = cfg['network']['handframe']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None

        self.mode = 'test' if mode == 'val' else mode
        self.kind = kind

        self.seq_lst = []
        self.fID_lst = []
        self.seq_start = []
        self.start_frame_lst = []
        self.mano_layer_right = OurManoLayer()
        test_data_dict = {}

        # Load sequence information from the split file
        for category in self.category_lst:
            split_file_name = 'finalv2_test_%s.npy' % category
            split_file_pth = pjoin(self.root_dset, 'splits', split_file_name)
            tmp_dict = np.load(split_file_pth, allow_pickle=True).item()
            for key, value in tmp_dict.items():
                test_data_dict[key] = value
        
        # Load data in sequence setting or single frame setting
        if self.kind == 'seq':
            for seq in test_data_dict.keys():
                for segment in test_data_dict[seq].keys():
                    idx_lst = test_data_dict[seq][segment]
                    self.seq_start.append(len(self.fID_lst))
                    self.seq_lst.extend([seq] * len(idx_lst))
                    self.fID_lst.extend(idx_lst)
                    self.start_frame_lst.extend([idx_lst[0]] * len(idx_lst))
            self.seq_start.append(len(self.fID_lst))
        elif self.kind == 'single_frame':
            for seq in test_data_dict.keys():
                for segment in test_data_dict[seq].keys():
                    idx_lst = test_data_dict[seq][segment]
                    self.seq_lst.extend([seq] * len(idx_lst))
                    self.fID_lst.extend(idx_lst)
            if self.mode == 'test':
                self.seq_lst = self.seq_lst[:64]
        self.len = len(self.seq_lst)
        print('HO3D mode %s %s: %d frames, use augment: %s' % (self.mode, self.kind, self.len, self.augment))
        
    def __getitem__(self, index):
        seq = self.seq_lst[index]
        fID = '%04d' % self.fID_lst[index]
        start_frame = self.start_frame_lst[index] if self.load_pred_obj_pose else None 
        cur_frame = self.fID_lst[index]

        data_dir = os.path.join(self.root_dset, 'train', '%s'% seq, 'pcd_raw')
        path = os.path.join(data_dir, fID + '.npy')

        full_data = generate_HO3D_data(self.mano_layer_right, self.root_dset, path, self.cfg['num_points'], self.cfg['obj_jitter_cfg'],
                                            self.cfg['hand_jitter_cfg'],
                                            self.cfg['device'], 
                                            load_pred_obj_pose=self.load_pred_obj_pose, 
                                            pred_obj_pose_dir=self.pred_obj_pose_dir,
                                            start_frame=start_frame,
                                            cur_frame=cur_frame,
                                            points_source=self.points_source,
                                            handframe=self.handframe)
        return full_data

    def __len__(self):
        return self.len


def visualize_data(data_dict, category):
    from vis_utils import plot3d_pts

    mano_pose = data_dict['gt_hand_pose']['mano_pose']
    trans = data_dict['gt_hand_pose']['translation']
    beta = data_dict['gt_hand_pose']['beta']
    mano_layer_right = OurManoLayer()
    hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                                                th_trans=torch.FloatTensor(trans).reshape(1, -1), th_betas=torch.from_numpy(beta).unsqueeze(0))
    hand_vertices = hand_vertices.cpu().data.numpy()[0]
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation']) * 5
    input_pc = np.matmul(data_dict['points'] - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices], [input_pc], [hand_vertices, input_pc]],
               show_fig=False, save_fig=True,
               save_folder=pjoin('HO3D', category),
               save_name=data_dict['file_name'])
