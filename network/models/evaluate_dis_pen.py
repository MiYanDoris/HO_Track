#evaluate distance and penetration

import trimesh
import numpy as np
import os
from os.path import join as pjoin
import pickle 
from manopth.manolayer import ManoLayer
import torch 
import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', type=str, default='bottle_sim')
parser.add_argument('-s', '--split', type=int, default=255)
parser.add_argument('-e', '--exp', type=str, default='1.20_baseline_newiknet_test')
args = parser.parse_args()

folder = f'/mnt/data/hewang/h2o_data/prediction/{args.exp}/results'
category = args.category
test_split = args.split

lst = os.listdir(folder)
lst = [i for i in lst if int(i.split('_')[-2]) >= test_split]
lst.sort()
lst = lst[:50]

mano_layer_right = ManoLayer(
        mano_root='/home/hewang/jiayi/manopth/mano/models', side='right', use_pca=False, ncomps=45,
        flat_hand_mean=True).cuda()

penetrate_num_pred = 0
avg_penetrate_dis_pred = 0
avg_tips_dis_pred = 0
avg_penetrate_point_pred = 0

penetrate_num_gt = 0
avg_penetrate_dis_gt = 0
avg_penetrate_point_gt = 0
avg_tips_dis_gt = 0

tips_index = [4,8,12,16,20]

for pkl in tqdm.tqdm(lst):
    path = pjoin(folder, pkl)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    instance = str(data['file_name'][0][0][:5])
    obj_mesh_path = f'/mnt/data/hewang/h2o_data/sim_onlygrasp/objs/{category}/{instance}.obj'
    obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
   # ppp = trimesh.proximity.ProximityQuery(obj_mesh)

    for i in range(50, 100):
        pred_kp = data['pred_hand_kp'][i][0,:,:]
        gt_hand_pose = data['gt_hand_poses'][i]
        pred_mano_pose = data['pred_hand_pose'][i]
        pred_mano_t = data['pred_hand_t'][i]
        gt_hand, gt_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(gt_hand_pose['mano_pose']).cuda(),
                                            th_trans=torch.FloatTensor(gt_hand_pose['mano_trans']).cuda())
        gt_kp = gt_kp.cpu().numpy()[0] / 1000
        gt_hand = gt_hand.cpu().numpy()[0] / 1000

        pred_hand, pred_kp_from_mano = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(pred_mano_pose).cuda(),
                                            th_trans=(torch.FloatTensor(pred_mano_t)).cuda())
        pred_hand = pred_hand.cpu().numpy()[0] / 1000
        hand_faces = mano_layer_right.th_faces.cpu().numpy()

        gt_obj_pose = {key:data['gt_obj_poses'][i][key] for key in data['gt_obj_poses'][i].keys()}

        pred_kp_objframe = np.matmul(pred_kp - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())
        pred_hand_objframe = np.matmul(pred_hand - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())
        gt_kp_objframe = np.matmul(gt_kp - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())
        gt_hand_objframe = np.matmul(gt_hand - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())
        
        hand_m = trimesh.Trimesh(vertices=pred_hand_objframe, faces=hand_faces)
        tri_coll = trimesh.collision.CollisionManager()
        tri_coll.add_object('obj',obj_mesh)
        tri_coll.add_object('hand',hand_m)
        coll_f, contact_lst = tri_coll.in_collision_internal(return_data=True)
        if coll_f:
            depth = np.max([xx.depth for xx in contact_lst])  
            if depth > 0.003:
                penetrate_num_pred += 1
                avg_penetrate_dis_pred += depth 
                avg_penetrate_point_pred += len(contact_lst)
        
        hand_m = trimesh.Trimesh(vertices=gt_hand_objframe, faces=hand_faces)
        tri_coll = trimesh.collision.CollisionManager()
        tri_coll.add_object('obj',obj_mesh)
        tri_coll.add_object('hand',hand_m)
        coll_f, contact_lst = tri_coll.in_collision_internal(return_data=True)
        if coll_f:
            depth = np.max([xx.depth for xx in contact_lst])  
            if depth > 0:
                penetrate_num_gt += 1
                avg_penetrate_dis_gt += depth 
                avg_penetrate_point_gt += len(contact_lst)
        # wrong!
        # pred_sd = -ppp.signed_distance(pred_kp_objframe)
        # gt_sd = -ppp.signed_distance(gt_kp_objframe)
        # if '00256_1' in data['file_name'][0][0]:
        #     print(pred_kp_objframe[12], pred_sd[12])
        # penetrate_num_pred += (pred_sd < -0.001).sum()
        # penetrate_num_gt += (gt_sd < -0.001).sum()
        # avg_penetrate_pred += ((pred_sd < -0.001) * pred_sd).sum()
        # avg_penetrate_gt += ((gt_sd < -0.001) * gt_sd).sum()
        # avg_tips_dis_pred += pred_sd[tips_index]
        # avg_tips_dis_gt += gt_sd[tips_index]
        
avg_penetrate_point_pred /= penetrate_num_pred
avg_penetrate_dis_pred /= penetrate_num_pred
avg_penetrate_dis_gt /= penetrate_num_gt
avg_penetrate_point_gt /= penetrate_num_gt
avg_tips_dis_pred /= len(lst) * 50
avg_tips_dis_gt /= len(lst) * 50
results_str = f'penetrate_num_pred:{penetrate_num_pred}\npenetrate_num_gt:{penetrate_num_gt}\navg_penetrate_dis_pred:{avg_penetrate_dis_pred}\navg_penetrate_point_pred:{avg_penetrate_point_pred}\navg_penetrate_dis_gt:{avg_penetrate_dis_gt}\navg_penetrate_point_gt:{avg_penetrate_point_gt}\n'

print(results_str)
with open(pjoin(folder, '../log/evalaute_dis_pen.txt'), 'w') as f:
    f.write(results_str)