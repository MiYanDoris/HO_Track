#evaluate distance and penetration

import logging
import trimesh
import numpy as np
import os
from os.path import join as pjoin
import pickle 
from manopth.manolayer import ManoLayer
import torch 
import tqdm
import argparse
from pointnet_utils import knn_point, group_operation
import time
from configs.config import get_config
from DeepSDF import SDF
# from datasets.data_utils import farthest_point_sample
import matplotlib.pyplot as plt

penetrate_threshold = 0.003
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--category', type=str, default='bottle_sim')
parser.add_argument('-s', '--split', type=int, default=255)
parser.add_argument('-e', '--exp', type=str, default='2.4_handbase_newIKNet_gtSDF_CMAES_kpregu_vis20')
parser.add_argument('-m', '--start_from_mid', action='store_true')
parser.add_argument('-f', '--first_50', action='store_false')
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-g', '--gt', action='store_true')

parser.add_argument('--config', type=str, default='2.2_handbase_newIKNet_gtSDF_full_CMAES_test.yml')

args = parser.parse_args()

folder = f'/mnt/data/hewang/h2o_data/prediction/{args.exp}/results'
category = args.category
test_split = args.split
logger = logging.getLogger()
if args.debug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

lst = os.listdir(folder)
lst = [i for i in lst if int(i.split('_')[-2]) >= test_split]
lst.sort()
if args.first_50:
    lst = lst[:50]

mano_layer_right = ManoLayer(
        mano_root='/home/hewang/jiayi/manopth/mano/models', side='right', use_pca=False, ncomps=45,
        flat_hand_mean=True).cuda()

template_T = torch.tensor([[95.6699, 6.3834, 6.1863]]) / 1000

tips_index = [4,8,12,16,20]

bad_cases = ['00255_0','00256_1','00257_2','00262_0','00263_2','00266_1','00269_2','00270_0','00271_2']

cfg = get_config(args, save=False)
SDFDecode = SDF(cfg)

def sdf_estimator(hand_objframe, threshold=penetrate_threshold, save=False):
    _, penetrate_mask, pred_sdf = SDFDecode.get_penetrate_from_sdf(hand_objframe, threshold)
    penetrate_num = torch.sum(penetrate_mask)

    max_penetrate_depth = torch.clamp(-torch.min(pred_sdf), min=0)
    if save:
        np.savetxt('distance.txt', torch.cat([hand_objframe[0] * SDFDecode.normalization_scale, pred_sdf.transpose(-1, -2)], dim=-1).cpu().numpy())
        np.savetxt('mask.txt', torch.cat([hand_objframe[0] * SDFDecode.normalization_scale, penetrate_mask.transpose(-1, -2)], dim=-1).cpu().numpy())
        np.savetxt('distance_real.txt', torch.cat([hand_objframe[0], pred_sdf.transpose(-1, -2)], dim=-1).cpu().numpy())
        np.savetxt('mask_real.txt', torch.cat([hand_objframe[0], penetrate_mask.transpose(-1, -2)], dim=-1).cpu().numpy())

    return penetrate_num, max_penetrate_depth, penetrate_mask

def batch_mesh_contains_points(
    ray_origins,
    obj_triangles,
    direction=torch.Tensor([1.0, 0.0, 0.0]).cuda(),
):
    """Times efficient but memory greedy !
    Computes ALL ray/triangle intersections and then counts them to determine
    if point inside mesh

    Args:
    ray_origins: (batch_size x point_nb x 3)
    obj_triangles: (batch_size, triangle_nb, vertex_nb=3, vertex_coords=3)
    Returns:
    exterior: (batch_size, point_nb) 1 if the point is outside mesh, 0 else
    """
    tol_thresh = 0.0000001

    batch_size = obj_triangles.shape[0]
    triangle_nb = obj_triangles.shape[1]
    point_nb = ray_origins.shape[1]

    # Batch dim and triangle dim will flattened together
    batch_points_size = batch_size * triangle_nb
    # Direction is random but shared
    v0, v1, v2 = obj_triangles[:, :, 0], obj_triangles[:, :, 1], obj_triangles[:, :, 2] # B, t, 3
    # Get edges
    v0v1 = v1 - v0
    v0v2 = v2 - v0

    # Expand needed vectors
    batch_direction = direction.view(1, 1, 3).expand(batch_size, triangle_nb, 3)

    # Compute ray/triangle intersections
    pvec = torch.cross(batch_direction, v0v2, dim=2) # B, t
    dets = torch.bmm(
        v0v1.view(batch_points_size, 1, 3), pvec.view(batch_points_size, 3, 1)
    ).view(batch_size, triangle_nb)

    # Check if ray and triangle are parallel
    parallel = abs(dets) < tol_thresh
    invdet = 1 / (dets + 0.1 * tol_thresh)

    # Repeat mesh info as many times as there are rays
    triangle_nb = v0.shape[1]
    v0 = v0.repeat(1, point_nb, 1) # B, t*p, 3
    v0v1 = v0v1.repeat(1, point_nb, 1)
    v0v2 = v0v2.repeat(1, point_nb, 1)
    hand_verts_repeated = (
        ray_origins.view(batch_size, point_nb, 1, 3)
        .repeat(1, 1, triangle_nb, 1)
        .view(ray_origins.shape[0], triangle_nb * point_nb, 3)
    ) # B, t*p, 3
    pvec = pvec.repeat(1, point_nb, 1) # B, tp
    invdet = invdet.repeat(1, point_nb)
    tvec = hand_verts_repeated - v0
    u_val = (
        torch.bmm(
            tvec.view(batch_size * tvec.shape[1], 1, 3),
            pvec.view(batch_size * tvec.shape[1], 3, 1),
        ).view(batch_size, tvec.shape[1])
        * invdet
    )
    # Check ray intersects inside triangle
    u_correct = (u_val > 0) * (u_val < 1)
    qvec = torch.cross(tvec, v0v1, dim=2)

    batch_direction = batch_direction.repeat(1, point_nb, 1)
    v_val = (
        torch.bmm(
            batch_direction.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    v_correct = (v_val > 0) * (u_val + v_val < 1)
    t = (
        torch.bmm(
            v0v2.view(batch_size * qvec.shape[1], 1, 3),
            qvec.view(batch_size * qvec.shape[1], 3, 1),
        ).view(batch_size, qvec.shape[1])
        * invdet
    )
    # points = ray_origins.reshape(1, 1, 3).repeat(1, 2864, 1) + t.reshape(1, 2864, 1) * direction.reshape(1, 1, 3).repeat(1, 2864, 1)
    # Check triangle is in front of ray_origin along ray direction
    t_pos = t >= tol_thresh
    parallel = parallel.repeat(1, point_nb)
    # # Check that all intersection conditions are met
    not_parallel = parallel.logical_not()
    final_inter = v_correct * u_correct * not_parallel * t_pos

    # np.savetxt('points.txt', points[0][final_inter[0]].cpu().numpy())
    # traingle_points = obj_triangles[final_inter]
    # np.savetxt('triangle_1.txt', traingle_points[0].cpu().numpy())
    # np.savetxt('triangle_2.txt', traingle_points[1].cpu().numpy())

    # Reshape batch point/vertices intersection matrix
    # final_intersections[batch_idx, point_idx, triangle_idx] == 1 means ray
    # intersects triangle
    final_intersections = final_inter.view(batch_size, point_nb, triangle_nb)
    # Check if intersection number accross mesh is odd to determine if point is
    # outside of mesh
    exterior = final_intersections.sum(2) % 2 == 0
    return exterior, final_intersections.sum(2)

def get_distance(hand, obj):
    return torch.min(torch.norm(hand.reshape(1, -1, 1, 3) - obj.reshape(1, 1, -1, 3), dim=-1), dim=-1)[0]

def ray_estimator(pred_hand_objframe, obj_triangles, obj_points):
    ray_penetrate_sum = torch.zeros((1, 778)).cuda()
    for i in range(6):
        theta = np.pi / 3 * i
        direction = torch.Tensor([np.cos(theta), 0.0, np.sin(theta)]).cuda()
        exterior, _ = batch_mesh_contains_points(pred_hand_objframe, obj_triangles, direction)
        ray_penetrate_sum += ~exterior
    ray_penetrate_mask = ray_penetrate_sum > 3

    distance = get_distance(pred_hand_objframe, obj_points)

    ray_max_penatrate = torch.max(ray_penetrate_mask * distance)
    penetrate_depth = ray_penetrate_mask * distance
    ray_penetrate_num = torch.sum(ray_penetrate_mask[penetrate_depth > penetrate_threshold])

    # np.savetxt('ray_mask.txt', torch.cat([pred_hand_objframe[0], ray_penetrate_mask[0].unsqueeze(1)], dim=-1).cpu().numpy())
    # np.savetxt('ray_distance.txt', torch.cat([pred_hand_objframe[0], distance[0].unsqueeze(1)], dim=-1).cpu().numpy())
    # exit(0)
    return ray_max_penatrate, ray_penetrate_num

ray_pred_num_lst = []
ray_pred_depth_lst = []
ray_gt_num_lst = []
ray_gt_depth_lst = []
sdf_pred_num_lst = []
sdf_pred_depth_lst = []
sdf_gt_num_lst = []
sdf_gt_depth_lst = []

if args.start_from_mid:
    start_frame = 50
else:
    start_frame = 0

for pkl in tqdm.tqdm(lst):
    penetrate_num_pred_ray = 0
    max_penetrate_depth_pred_ray = 0

    penetrate_num_pred_sdf = 0
    max_penetrate_depth_pred_sdf = 0

    penetrate_num_gt_ray = 0
    max_penetrate_depth_gt_ray = 0

    penetrate_num_gt_sdf = 0
    max_penetrate_depth_gt_sdf = 0

    print(pkl)
    path = pjoin(folder, pkl)
    with open(path, 'rb') as f:
        data = pickle.load(f)
    instance = str(data['file_name'][0][0][:5])
    SDFDecode.load_obj(data['file_name'][0][0][:11])
    # if str(data['file_name'][0][0][:7]) not in bad_cases:
    #     continue
    obj_mesh_path = f'/mnt/data/hewang/h2o_data/sim_onlygrasp/manifold_simplified_objs/{category}/{instance}.obj'
    obj_mesh = trimesh.load(obj_mesh_path, force='mesh')
    print('faces num: %d, vertices num: %d' % (obj_mesh.triangles.shape[0], obj_mesh.vertices.shape[0]))

    traingle_num = 300000
    if obj_mesh.triangles.shape[0] > traingle_num:
        stride = ((obj_mesh.triangles.shape[0] - 1) // traingle_num) + 1
    else:
        stride = 1
    obj_triangles = torch.FloatTensor(obj_mesh.triangles[::stride]).reshape(1, -1, 3, 3).cuda()

    obj_num = 200000
    if obj_mesh.vertices.shape[0] > obj_num:
        stride = ((obj_mesh.vertices.shape[0] - 1) // obj_num) + 1
    else:
        stride = 1
    obj_points = torch.FloatTensor(obj_mesh.vertices[::stride]).reshape(1, -1, 3).cuda()

    # obj_triangles = torch.FloatTensor(obj_mesh.triangles).reshape(1, -1, 3, 3).cuda()
    # obj_points = torch.FloatTensor(obj_mesh.vertices[::10]).reshape(1, -1, 3).cuda()

    for i in range(start_frame, 100):
        with torch.no_grad():
            pred_kp = data['pred_hand_kp'][i][0,:,:]
            gt_hand_pose = data['gt_hand_poses'][i]
            pred_mano_pose = data['pred_hand_pose'][i]
            pred_mano_t = data['pred_hand_t'][i]
            
            pred_hand, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(pred_mano_pose).cuda(),
                                                th_trans=torch.FloatTensor(pred_mano_t).cuda())
            pred_hand = pred_hand / 1000

            gt_obj_pose = {key:torch.FloatTensor(data['gt_obj_poses'][i][key]).cuda() for key in data['gt_obj_poses'][i].keys()}

            pred_hand_objframe = torch.matmul(pred_hand - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())

            ray_max_penatrate, ray_penetrate_num = ray_estimator(pred_hand_objframe, obj_triangles, obj_points)
            logging.debug('-' * 40)
            logging.debug('Ray penetrate num: %d' % ray_penetrate_num)
            logging.debug('Ray penetration_depth: %.05f mm' % (ray_max_penatrate * 1000))
            penetrate_num_pred_ray += ray_penetrate_num
            max_penetrate_depth_pred_ray += ray_max_penatrate

            penetrate_num, max_penetrate_depth, _ = sdf_estimator(pred_hand_objframe, save=False)
            penetrate_num_pred_sdf += penetrate_num
            max_penetrate_depth_pred_sdf += max_penetrate_depth
            logging.debug('SDF penetrate num: %d' % penetrate_num)
            logging.debug('SDF penetration_depth: %.05f mm' % (max_penetrate_depth * 1000))

            if args.gt:
                gt_hand, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(gt_hand_pose['mano_pose']).cuda(),
                                                    th_trans=torch.FloatTensor(gt_hand_pose['mano_trans']).cuda())
                gt_hand = gt_hand / 1000
                gt_hand_objframe = torch.matmul(gt_hand - gt_obj_pose['translation'].squeeze(), gt_obj_pose['rotation'].squeeze())


                ray_max_penatrate, ray_penetrate_num = ray_estimator(gt_hand_objframe, obj_triangles, obj_points)
                penetrate_num_gt_ray += ray_penetrate_num
                max_penetrate_depth_gt_ray += ray_max_penatrate

                penetrate_num, max_penetrate_depth, _ = sdf_estimator(gt_hand_objframe, save=False)
                penetrate_num_gt_sdf += penetrate_num
                max_penetrate_depth_gt_sdf += max_penetrate_depth

    results_str = 'ray estimator: \n\n'
    ray_mean_penetrate_num_pred = penetrate_num_pred_ray / (100 - start_frame)
    ray_mean_penetrate_depth_pred = max_penetrate_depth_pred_ray / (100 - start_frame)
    ray_mean_penetrate_num_gt = penetrate_num_gt_ray / (100 - start_frame)
    ray_mean_penetrate_depth_gt = max_penetrate_depth_gt_ray / (100 - start_frame)
    results_str += f'penetrate_num_pred:{ray_mean_penetrate_num_pred}\npenetrate_num_gt:{ray_mean_penetrate_num_gt}\nmax_penetration_pred:{ray_mean_penetrate_depth_pred}\nmax_penetration_gt:{ray_mean_penetrate_depth_gt}\n\n\n'

    results_str += 'SDF estimator: \n\n'
    sdf_mean_penetrate_num_pred = penetrate_num_pred_sdf / (100 - start_frame)
    sdf_mean_penetrate_depth_pred = max_penetrate_depth_pred_sdf / (100 - start_frame)
    sdf_mean_penetrate_num_gt = penetrate_num_gt_sdf / (100 - start_frame)
    sdf_mean_penetrate_depth_gt = max_penetrate_depth_gt_sdf / (100 - start_frame)
    results_str += f'penetrate_num_pred:{sdf_mean_penetrate_num_pred}\npenetrate_num_gt:{sdf_mean_penetrate_num_gt}\nmax_penetration_pred:{sdf_mean_penetrate_depth_pred}\nmax_penetration_gt:{sdf_mean_penetrate_depth_gt}\n\n'
    print(results_str)

    ray_pred_num_lst.append(ray_mean_penetrate_num_pred.cpu().numpy())
    ray_pred_depth_lst.append(ray_mean_penetrate_depth_pred.cpu().numpy())
    sdf_pred_num_lst.append(sdf_mean_penetrate_num_pred.cpu().numpy())
    sdf_pred_depth_lst.append(sdf_mean_penetrate_depth_pred.cpu().numpy())

    if args.gt:
        ray_gt_num_lst.append(ray_mean_penetrate_num_gt.cpu().numpy())
        ray_gt_depth_lst.append(ray_mean_penetrate_depth_gt.cpu().numpy())
        sdf_gt_num_lst.append(sdf_mean_penetrate_num_gt.cpu().numpy())
        sdf_gt_depth_lst.append(sdf_mean_penetrate_depth_gt.cpu().numpy())

ray_pred_num = np.mean(np.array(ray_pred_num_lst))
ray_pred_depth = np.mean(np.array(ray_pred_depth_lst))
sdf_pred_num = np.mean(np.array(sdf_pred_num_lst))
sdf_pred_depth = np.mean(np.array(sdf_pred_depth_lst))

if args.gt:
    ray_gt_num = np.mean(np.array(ray_gt_num_lst))
    ray_gt_depth = np.mean(np.array(ray_gt_depth_lst))
    sdf_gt_num = np.mean(np.array(sdf_gt_num_lst))
    sdf_gt_depth = np.mean(np.array(sdf_gt_depth_lst))

print('-' * 80)
print('Overall: ')
print(f'ray_pred_num = {ray_pred_num}\nray_pred_depth = {ray_pred_depth}\nsdf_pred_num = {sdf_pred_num}\nsdf_pred_depth = {sdf_pred_depth}\n')
if args.gt:
    print(f'ray_gt_num = {ray_gt_num}\nray_gt_depth = {ray_gt_depth}\nsdf_gt_num = {sdf_gt_num}\nsdf_gt_depth = {sdf_gt_depth}')
