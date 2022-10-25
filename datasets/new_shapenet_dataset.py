import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import torch
from tqdm import tqdm
import time 
from network.models.our_mano import OurManoLayer
from network.models.hand_utils import handkp2palmkp

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from configs.config import get_config
from data_utils import farthest_point_sample, mat_from_rvec, split_dataset, jitter_hand_mano, jitter_obj_pose, pose_list_to_dict, jitter_hand_kp, OBB

"""
ShapeNet data organization:
render/
    img/
        instance_hand_camera/
            depth.png
            mask.png
    preproc/
        xxx.npz
        ...
    meta.txt
"""

category2scale = {
    'bottle_sim': 0.25,
    'bowl_sim': 0.25,
    'car_sim': 0.3,
}



def generate_shapenet_data(path, category, num_parts, num_points, obj_perturb_cfg, hand_jitter_config, device, handframe, input_only, obj_path, skip_data, mano_layer_right, load_pred_obj_pose=False,pred_obj_pose_dir=None):
    '''
    input_only = 'hand' or 'obj' or 'both'
    skip_data = 'hand' or 'obj' or 'both'
    '''
    #read file
    
    cloud_dict = np.load(path, allow_pickle=True)["all_dict"].item()

    # if hand_jitter_config['jitter_shape']:
    #     mano_pose = np.array(cloud_dict['hand_pose']['mano_pose'])
    #     hand_global_rotation = mat_from_rvec(mano_pose[:3])
    #     mano_trans = np.array(cloud_dict['hand_pose']['mano_trans'])
    #     mano_beta = np.array(cloud_dict['hand_pose']['mano_beta'])
    #     pose = torch.FloatTensor(np.array(mano_pose).reshape(1, -1)).cuda()
    #     beta = (torch.randn(size=(1, 10))*np.sqrt(3))
    #     _, hand_kp = mano_layer_right.forward(th_pose_coeffs=pose, th_trans=torch.FloatTensor(mano_trans).reshape(1, -1).cuda(), th_betas=beta.cuda(), original_version=True)
    #     hand_kp = hand_kp[0].cpu()      #[21,3]
    #     world_trans = hand_kp[0]        
    #     rest_pose = pose.clone()
    #     rest_pose[:, :3] = 0
    #     _, template_kp = mano_layer_right.forward(th_pose_coeffs=rest_pose, th_trans=(torch.zeros((1, 3))).cuda(), th_betas=beta.cuda())
    #     palm_template = handkp2palmkp(template_kp)
    #     palm_template = palm_template[0].float().cpu()

    #     jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)
    #     hand_kp = hand_kp.numpy()
    #     jittered_hand_kp = jittered_hand_kp.numpy()

    #     full_data = {
    #         'jittered_hand_kp': jittered_hand_kp,
    #         'gt_hand_kp': hand_kp,
    #         'gt_hand_pose':{'translation':np.expand_dims(world_trans, axis=1),
    #                         'scale': 0.2,
    #                         'rotation': np.array(hand_global_rotation),
    #                         'mano_pose':mano_pose,
    #                         'mano_trans':mano_trans,
    #                         'palm_template': palm_template,
    #                         'mano_beta': beta[0],
    #                         },
    #         'category': category,
    #         'file_name':cloud_dict['file_name'],
    #     }

    #     return full_data 

    cam = cloud_dict['points']
    label = cloud_dict['labels']
    if len(cam) == 0:
        return None

    #shuffle
    n = cam.shape[0]
    perm = np.random.permutation(n)
    cam = cam[perm]
    label = label[perm]

    #filter
    hand_id = num_parts
    hand_idx = np.where(label == hand_id)[0]
    if len(hand_idx) == 0:
        return None
    else:
        hand_pcd = cam[hand_idx]
        sample_idx = farthest_point_sample(hand_pcd, num_points, device)
        hand_pcd = hand_pcd[sample_idx]

    obj_idx = np.where(label != hand_id)[0]
    if len(obj_idx) == 0:
        return None 
    else:
        obj_pcd = cam[obj_idx]
        sample_idx = farthest_point_sample(obj_pcd, num_points, device)
        obj_pcd = obj_pcd[sample_idx]

    if input_only == 'hand':
        idx = np.where(label == hand_id)[0]
        if len(idx) == 0:
            return None
        cam = cam[idx]
        label = label[idx]
    elif input_only == 'obj':
        idx = np.where(label != hand_id)[0]
        if len(idx) == 0:
            return None
        cam = cam[idx]
        label = label[idx]

    # sampling
    sample_idx = farthest_point_sample(cam, num_points, device)
    seg = label[sample_idx]
    cam_points = cam[sample_idx]
    if cam_points is None:
        return None
    #generate obj canonical point clouds
    obj_pose = cloud_dict['obj_pose']
    if num_parts == 1:
        obj_pose = [obj_pose]
    obj_nocs = np.zeros_like(cam_points)

    for i in range(num_parts):
        obj_idx = np.where(seg == i)[0]
        if len(obj_idx)!=0:
            obj_nocs[obj_idx] = np.matmul((cam_points[obj_idx] - np.expand_dims(obj_pose[i]['translation'], 0)) / obj_pose[i]['scale'],
                                      obj_pose[i]['rotation'])
        obj_pose[i]['translation'] = np.expand_dims(np.array(obj_pose[i]['translation']), axis=1)

    obj_idx = np.where(seg != hand_id)[0]
    if len(obj_idx)==0:
      #  print('%s has no object!'%(path))
        no_obj_flag = True
    else:
        no_obj_flag = False
    #generate hand canonical point clouds
    mano_pose = np.array(cloud_dict['hand_pose']['mano_pose'])
    hand_global_rotation = mat_from_rvec(mano_pose[:3])
    mano_trans = np.array(cloud_dict['hand_pose']['mano_trans'])
    mano_beta = np.array(cloud_dict['hand_pose']['mano_beta'])
    hand_idx = np.where(seg == hand_id)[0]
    if len(hand_idx)==0:
        no_hand_flag = True
    else:
        no_hand_flag = False

    #skip data
    if input_only == 'both':
        if skip_data == 'hand' and no_hand_flag:
            return None
        if skip_data == 'obj' and no_obj_flag:
            return None
        if skip_data == 'both' and (no_hand_flag or no_obj_flag):
            return None

    hand_template, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)).cuda(),
                                            th_trans=torch.FloatTensor(mano_trans).reshape(1, -1).cuda(),
                                            th_betas=torch.FloatTensor(mano_beta).reshape(1, -1).cuda(), original_version=True)
    # hand_template = hand_template[0].cpu()
    beta = mano_beta.reshape(1,-1)
    rest_pose = np.zeros_like(mano_pose)
    hand_kp = hand_kp[0].cpu()
    world_trans = hand_kp[0] 
    hand_kp = hand_kp.numpy()

    _, template_kp = mano_layer_right.forward(th_pose_coeffs=(torch.zeros((1, 48))).cuda(), th_trans=(torch.zeros((1, 3))).cuda(),
                                            th_betas=torch.FloatTensor(mano_beta).reshape(1, -1).cuda())
    palm_template = handkp2palmkp(template_kp)
    palm_template = palm_template[0].float().cpu()

    #jitter hand pose and obj pose
    # if hand_jitter_config['jitter_by_MANO']:
    #     jittered_mano, jittered_trans, jittered_beta = jitter_hand_mano(torch.FloatTensor(hand_global_rotation), np.array(mano_pose[3:]), mano_trans, mano_beta, hand_jitter_config)
    #     _, jittered_hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(jittered_mano.reshape(1, -1)).cuda(),
    #                                             th_trans=torch.FloatTensor(jittered_trans).reshape(1, -1).cuda(),
    #                                             th_betas=torch.FloatTensor(jittered_beta).reshape(1, -1).cuda(), original_version=True)
    #     jittered_hand_kp = jittered_hand_kp[0].cpu().numpy()
    # else:
    jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)

    pose_perturb_cfg = {'type': obj_perturb_cfg['type'],
                        'scale': obj_perturb_cfg['s'],
                        'translation': obj_perturb_cfg['t'],  # Now pass the sigma of the norm
                        'rotation': np.deg2rad(obj_perturb_cfg['r'])}
    jittered_obj_pose_lst = []
    for i in range(num_parts):
        jittered_obj_pose = jitter_obj_pose(obj_pose[i], pose_perturb_cfg)
        jittered_obj_pose_lst.append(jittered_obj_pose)

    if 'sim_dataset' in path:
        grasptime = cloud_dict['grasptime']
    else:
        grasptime = 0
    
    # distance = np.min(np.linalg.norm(hand_pcd.reshape(-1, 1, 3) - hand_template.reshape(1, -1, 3), axis=-1), axis=0)
    # visibility = distance < visible_thresh
    
    full_data = {
        'hand_points': hand_pcd,
        'obj_points': obj_pcd,
        'points': cam_points,
        'labels': seg,  #hand-1 object-0
        'obj_nocs': obj_nocs,
        'hand_id': hand_id,
        'jittered_obj_pose': pose_list_to_dict(jittered_obj_pose_lst),     # list
        'gt_obj_pose': pose_list_to_dict(obj_pose),                        # list
        'jittered_hand_kp': jittered_hand_kp,
        'gt_hand_kp': hand_kp,
        # 'jittered_hand_template': jittered_hand_template,
        'gt_hand_pose':{'translation':np.expand_dims(world_trans, axis=1),
                          'scale': 0.2,
                          'rotation': np.array(hand_global_rotation),
                          'mano_pose':mano_pose,
                          'mano_trans':mano_trans,
                          'palm_template': palm_template,
                          'mano_beta': beta[0],
                        #   'hand_template': hand_template,
                        #   'visibility': visibility
                          },
        'category': category,
        'file_name':cloud_dict['file_name'],
        'static_flag': int(cloud_dict['file_name'].split('_')[-1]) >= grasptime,
        'mesh_path': '/mnt/data/hewang/h2o_data/sim_dataset/objs/%s/%s.obj' % (category, cloud_dict['file_name'].split('_')[0]),
        'projection': { 'cx': 512/2, 'cy': 424/2, 'fx': -1.4343544 * 512/ 2.0, 'fy': 1.7320507 * 424 / 2.0, 'h': 424, 'w': 512}
    }
    full_data['gt_obj_pose']['up_and_down_sym'] = False
    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s_%s.pkl'%(category, path.split('/')[-1][:-8]))
        tmp = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_dict = tmp
        del tmp
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = int(path.split('/')[-1][-7:-4])
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose
    if handframe == 'OBB':
        _,full_data['OBB_pose'] = OBB(cam_points) 
        if full_data['OBB_pose']['scale'] < 0.001:
            return None 
    if obj_path is not None:
        #print(cloud_dict['file_name'], obj_path)
        full_obj_nocs = np.load(obj_path)
        idx = np.random.permutation(4096)
        full_obj_nocs = full_obj_nocs[idx]
        full_data['full_obj'] = full_obj_nocs[farthest_point_sample(full_obj_nocs, 1024, device)] 
        full_data['full_obj'] = np.matmul(full_data['full_obj']* category2scale[category], obj_pose[0]['rotation'].transpose(-1,-2)) + obj_pose[0]['translation'].transpose(-1,-2)
    # for iknet
    # if load_baseline:
    #     pred_path = path.replace('render/preproc', 'baseline_prediction')
    #     pred_path = pred_path.replace('seq', 'single_frame')
    #     pred_path = pred_path.replace('.npz', '.npy')
    #     if not os.path.isfile(pred_path):
    #         print('No pred path! ', pred_path)
    #         return None
    #     pred_dict = np.load(pred_path, allow_pickle=True).item()
    #     full_data['refine_dict'] = pred_dict
    return full_data



class NewSHAPENETDataset:
    def __init__(self, cfg, mode, kind):
        '''
        kind: 'single_frame' or 'seq'
        mode: use 'test' to replace 'val'
        '''
        self.cfg = cfg
        self.root_dset = cfg['data_cfg']['basepath']
        self.obj_cat_lst = cfg['obj_category']
        self.load_baseline = True if cfg['network']['type'] == 'iknet' else False # laod baseline
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']
        self.handframe = cfg['network']['handframe']
        # self.visible_thresh = cfg['visible_thresh']
        self.kind = ['seq']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None

        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right').cuda()
        self.file_list = []
        self.num_parts = {}
        if mode == 'val':
            mode = 'test'
        for cat in self.obj_cat_lst:
            for k in self.kind:
                self.num_parts[cat] = self.cfg['data_cfg'][cat]['num_parts']
                read_folder = pjoin(self.root_dset, 'preproc', cat, k)
                splits_folder = pjoin(self.root_dset, "splits", cat, k)
                use_txt = pjoin(splits_folder, f"{mode}.txt")
                splits_ready = os.path.exists(use_txt)
                if not splits_ready:
                    if 'train_val_split' in self.cfg['data_cfg'][cat]:
                        split = self.cfg['data_cfg'][cat]['train_val_split']
                        train_ins_lst = ['%05d' % i for i in range(split[0])]
                        test_ins_lst = ['%05d' % i for i in range(split[0], split[0] + split[1])]
                    else:
                        train_ins_lst = None
                        test_ins_lst = self.cfg['data_cfg'][cat]['test_list']
                    split_dataset(splits_folder, read_folder, test_ins_lst, train_ins_lst)
                with open(use_txt, "r", errors='replace') as fp:
                    lines = fp.readlines()
                    # if len(lines) % 100 != 0:
                    #     print(use_txt)
                    #     ins_lst = list(set([i[:5] for i in lines]))
                    #     for ins in ins_lst:
                    #         s = np.sum([1 for i in lines if ins in i])
                    #         if s % 100 != 0:
                    #             print(ins)
                    file_list = [pjoin(read_folder, i.strip()) for i in lines]
                self.file_list.extend(file_list)

        self.len = len(self.file_list)
        # if self.cfg['gt_or_recon'] == 'recon':
        #     self.recon_dict = {}
        #     for cat in self.obj_cat_lst:
        #         self.recon_dict[cat] = {}
        #         recon_folder = pjoin(self.root_dset, '..', 'reconstruction', cat, 'pc')
        #         lst = os.listdir(recon_folder)
        #         ins_lst = list(set([i.split('_')[0] for i in lst]))
        #         for ins in ins_lst:
        #             self.recon_dict[cat][ins] = []
        #         for path in lst:
        #             self.recon_dict[cat][path.split('_')[0] ].append(pjoin(recon_folder, path))
        print(f"mode: {mode}, kind: {self.kind}, data number: {self.len}, obj_lst: {self.obj_cat_lst}")

    def __getitem__(self, index):
        path = self.file_list[index]
        ins = self.file_list[index].split('/')[-1].split('_')[0]
        category = self.file_list[index].split('/')[-3]
        num_parts = self.num_parts[category]
        if self.cfg['add_obj'] and self.cfg['gt_or_recon'] == 'recon' and ins not in self.recon_dict[category].keys():
            print(category, ' has no reconstruction ', ins)
            full_data = None
        else:
            if self.cfg['add_obj']:
                if self.cfg['gt_or_recon'] == 'recon':
                    if self.kind == ['seq']:
                        #use the same instance while tracking!
                        obj_path = self.recon_dict[category][ins][0]
                    else:
                        num = len(self.recon_dict[category][ins])
                        rand_num = np.random.randint(0,num)
                        obj_path = self.recon_dict[category][ins][rand_num]
                elif self.cfg['gt_or_recon'] == 'gt':
                    obj_path = pjoin(self.root_dset, '..', 'full_nocs_pc', category, ins+'.npy')
                else:
                    raise NotImplementedError
            else:
                obj_path = None

            full_data = generate_shapenet_data(
                                                path, 
                                                category, 
                                                num_parts, 
                                                self.cfg['num_points'], 
                                                self.cfg['obj_jitter_cfg'],
                                                self.cfg['hand_jitter_cfg'],
                                                self.cfg['device'], 
                                                self.handframe,
                                                input_only=self.cfg['input_only'],
                                                obj_path=obj_path, 
                                                skip_data=self.cfg['skip_data'], 
                                                mano_layer_right=self.mano_layer_right, 
                                                load_pred_obj_pose=self.load_pred_obj_pose,
                                                pred_obj_pose_dir=self.pred_obj_pose_dir,
                                                )
        
        return full_data

    def __len__(self):
        return self.len


def visualize_data(data_dict, category):
    from vis_utils import plot3d_pts

    # import torch
    # full_obj = data_dict['full_obj']

    # mano_pose = data_dict['gt_hand_pose']['mano_pose']
    # mano_trans = data_dict['gt_hand_pose']['mano_trans']
    # mano_layer_right = ManoLayer(
    #         mano_root=mano_root , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    # hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
    #                                             th_trans=torch.FloatTensor(mano_trans).reshape(1, -1))
    # hand_vertices = hand_vertices.cpu().data.numpy()[0] / 1000
    hand_vertices = data_dict['gt_hand_pose']['hand_template']
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    visibility = data_dict['gt_hand_pose']['visibility']
    visible_template = hand_vertices[visibility]
    invisible_template = hand_vertices[~visibility]

    hand_pcd = data_dict['hand_points']
    obj_pcd = data_dict['obj_points']
    hand_pcd = np.matmul(hand_pcd - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    obj_pcd = np.matmul(obj_pcd - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5

    # full_obj = np.matmul(full_obj - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices, obj_pcd], [hand_pcd, obj_pcd], [visible_template], [invisible_template]],
               show_fig=False, save_fig=True,
               save_folder=pjoin('shapenet_data_vis', category),
               save_name=data_dict['file_name'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='2.28_hand_joint.yml', help='path to config.yml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--kind', type=str, default='single_frame', choices=['single_frame', 'seq'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args, save=False)
    dataset = NewSHAPENETDataset(cfg, args.mode, args.kind)
    print(len(dataset))
    mano_diff_lst = []
    trans_diff_lst = []

    for i in range(5):
        data_dict = dataset[i * 2000]
        visualize_data(data_dict, 'bottle')

    # N = 5000
    # for i in tqdm(range(len(dataset) // N)):
    #     for j in tqdm(range(100)):
    #         mano = dataset[N * i + j]['gt_hand_pose']['mano_pose']
    #         translation = dataset[N * i + j]['gt_hand_pose']['mano_trans']
    #         if j > 0:
    #             trans_diff_lst.append(translation - last_translation)
    #             mano_diff_lst.append(mano - last_mano)

    #         last_translation = translation
    #         last_mano = mano
    
    # trans_array = np.array(trans_diff_lst)
    # print(np.max(trans_array, axis=0))
    # print(np.min(trans_array, axis=0))

    # mano_array = np.array(mano_diff_lst)
    # print(np.max(mano_array, axis=0))
    # print(np.min(mano_array, axis=0))

    # np.save('hand_trans.npy', trans_array)
    # np.save('hand_mano.npy', mano_array)
    
    ''' # examine MANO PCA
    import pickle
    import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--category', type=str, default='bottle_sim')
    parser.add_argument('-s', '--split', type=int, default=255)
    parser.add_argument('-e', '--exp', type=str, default='1.25_baseline_newiknet_optimize_test')
    args = parser.parse_args()

    folder = f'/mnt/data/hewang/h2o_data/prediction/{args.exp}/results'
    category = args.category
    test_split = args.split

    lst = os.listdir(folder)
    lst = [i for i in lst if int(i.split('_')[-2]) >= test_split]
    lst.sort()
    lst = lst[:50]
    
    ncomps = 6
    mano_layer_right = ManoLayer(
        mano_root='/home/hewang/jiayi/manopth/mano/models', side='right', use_pca=True, ncomps=ncomps,
        flat_hand_mean=True)
    
    coeff = mano_layer_right.th_selected_comps
    transform = torch.matmul(coeff.T, coeff)
    
    error_lst = []
    for pkl in lst:
        path = pjoin(folder, pkl)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gt_mano = torch.zeros((100, 45))
        for i in range(100):
            gt_hand_pose = data['gt_hand_poses'][i]
            gt_mano[i] = torch.FloatTensor(gt_hand_pose['mano_pose'][0][3:])
        transformed_mano = torch.matmul(gt_mano, transform)
        error = (transformed_mano - gt_mano).abs().mean()
        error_lst.append(error)
    print(np.mean(np.array(error_lst)))

    # hand_vertices, kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(mano_rest).unsqueeze(0))
    # print(torch.matmul(mano_rest, transform))
    '''
        
    
