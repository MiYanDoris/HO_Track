from datasets.real_dataset.loadfile import path_list2plainobj_input_check
import numpy as np
from network.models.hand_utils import handkp2palmkp
import os
from manopth.manolayer import ManoLayer
import pickle
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rt
import tqdm

mano_layer_right = ManoLayer(
            mano_root='/home/hewang/jiayi/manopth/mano/models' , side='right', use_pca=False, ncomps=45, flat_hand_mean=True).cuda()

def copy_objpose_to_objpose_refined(task_list):
    for task in task_list:
        obj_pose_lst = os.listdir(os.path.join(task, 'objpose'))
        instance_lst = [int(i.split('.')[0]) for i in obj_pose_lst]
        instance_lst.sort()
        for instance in instance_lst:
            os.system('cp %s %s' % (os.path.join(task, 'objpose', '%d.json' % instance), os.path.join(task, 'objpose_refined', '%04d.json' % instance)))

def get_hand_anno(data_dir, instance, name):
    pkl_path = os.path.join(data_dir, 'hand_pose_refined/%d.pickle' % instance)
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    mano_para = data['poseCoeff']
    mano_beta = data['beta']
    trans = data['trans']

    vertices, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(mano_para).unsqueeze(0).cuda(),
                                                    th_trans=torch.zeros((1, 3)).cuda(),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10).cuda()
                                                    )
    vertices = (vertices[0]/1000).cpu()
    hand_kp = (hand_kp[0]/1000).cpu()
    vertices += trans
    hand_kp += trans

    _, template_kp = mano_layer_right.forward(th_pose_coeffs=torch.zeros((1, 48)).cuda(),
                                                    th_trans=torch.zeros((1, 3)).cuda(),
                                                    th_betas=torch.FloatTensor(mano_beta).reshape(1, 10).cuda())
    template_kp = template_kp.cpu()
    template_trans = (template_kp[0] / 1000)[0].numpy()
    global_trans = trans + template_trans

    palm_template = handkp2palmkp(template_kp)[0] / 1000
    palm_template -= template_trans

    # hand_faces = mano_layer_right.th_faces.cpu().data.numpy()
    # mesh = trimesh.Trimesh(vertices=vertices, faces=hand_faces)
    # mesh_txt = trimesh.exchange.obj.export_obj(mesh, include_normals=False, include_color=False,
    #                                             include_texture=False, return_texture=False, write_texture=False,
    #                                             resolver=None, digits=8)
    # with open(name, "w") as fp:
    #     fp.write(mesh_txt)
    return vertices.numpy(), mano_para, mano_beta, trans, hand_kp.numpy(), global_trans, palm_template.numpy()

def generate_npy(task_dir, instance, camera_id, save_color_pcld=False, hand_pose=False):
    cam_in_path = '/mnt/data/hewang/h2o_data/HOI4D/camera_params/ZY2021080000%d/color_intrin.npy' % camera_id

    dpt_path = os.path.join(task_dir, 'align_depth/%d.png' % instance)
    mask_path = os.path.join(task_dir, '2Dseg/mask/%05d.png' % instance)

    anno_path = os.path.join(task_dir, 'objpose/%05d.json' % instance)
    if not os.path.exists(anno_path):
        anno_path = os.path.join(task_dir, 'objpose/%d.json' % instance)
        if not os.path.exists(anno_path):
            anno_path = os.path.join(task_dir, 'objpose_refined/%04d.json' % instance)
            if not os.path.exists(anno_path):
                return
    save_dir = os.path.join(task_dir, 'preprocess')
    save_path = os.path.join(save_dir, '%03d.npy' % instance)

    # if os.path.exists(save_path):
    #     return

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if save_color_pcld:
        color_pcld_save_path = os.path.join(task_dir, 'obj_pcld_vis/%03d.pcd' % instance)
        color_pcld_dir = os.path.dirname(color_pcld_save_path)
        if not os.path.exists(color_pcld_dir):
            os.mkdir(color_pcld_dir)
    else:
        color_pcld_save_path = None

    if hand_pose:
        vertices, mano_para, mano_beta, hand_trans, hand_kp, global_trans, palm_template = get_hand_anno(task_dir, instance, name=(task_dir + '/hand_mesh/hand_mesh_%d.obj' % instance))
    else:
        vertices=None

    rotation_mat, obj_trans, obj_mask, hand_mask, depth2d, obj_pcd, hand_pcd, camMat, crop_list, dim = path_list2plainobj_input_check(cam_in_path=cam_in_path, dpt_path=dpt_path, mask_path=mask_path, anno_path=anno_path, hand_vertices=vertices, denoise=True, color_pcld_save_path=color_pcld_save_path)

    data_dict = {
        'obj_pcld':obj_pcd,
        'hand_pcld':hand_pcd,
        'obj_pose': {
            'rotation': rotation_mat,
            'translation': obj_trans,
            'bbox_length': dim,
        },
        'projection': {
            'cx': camMat[0, 2],
            'cy': camMat[1, 2],
            'fx': camMat[0, 0],
            'fy': camMat[1, 1],
            'h': 1080,
            'w': 1920,
        }
    }
    if hand_pose:
        data_dict['hand_pose'] = {
            'hand_kp': hand_kp,
            'theta': mano_para,
            'beta': mano_beta,
            'mano_trans': hand_trans, # mano translation
            'global_trans': global_trans,
            'palm_template': palm_template
        }
    np.save(save_path, data_dict)

root_dir = '/mnt/data/hewang/h2o_data/HOI4D/C5'
# obj_lst = os.listdir(root_dir)
# obj_lst.sort()
# for obj_id in tqdm.tqdm(obj_lst):
#     obj_dir = os.path.join(root_dir, obj_id)
#     if 'txt' in obj_dir:
#         continue
#     scene_lst = os.listdir(obj_dir)
#     scene_lst.sort()
#     for scene in scene_lst:
#         scene_dir = os.path.join(obj_dir, scene)
#         layout_lst = os.listdir(scene_dir)
#         layout_lst.sort()
#         for layout in layout_lst:
#             layout_dir = os.path.join(scene_dir, layout)
#             task_lst = os.listdir(layout_dir)
#             task_lst.sort()
#             for task in task_lst:
#                 task_dir = os.path.join(layout_dir, task)
#                 for i in range(30):
#                     instance = i * 10
#                     generate_npy(task_dir=task_dir, instance=instance, camera_id=2, save_color_pcld=True, hand_pose=False)

# for refined object pose
task_list = []
txt_pth = '/mnt/data/hewang/h2o_data/HOI4D/C5/bottle_refined_pose.txt'
with open(txt_pth, "r", errors='replace') as fp:
    lines = fp.readlines()
    for i in lines:
        _, obj_id, scene, layout, task = i.strip().split('/')
        task_list.append(os.path.join(root_dir, obj_id, scene, layout, task))

for task in tqdm.tqdm(task_list):
    for instance in tqdm.tqdm(range(300)):
        if instance % 150 == 0:
            generate_npy(task_dir=task, instance=instance, camera_id=2, save_color_pcld=True, hand_pose=True)
        else:
            generate_npy(task_dir=task, instance=instance, camera_id=2, save_color_pcld=False, hand_pose=True)

# task = '/mnt/data/hewang/h2o_data/HOI4D/C5/N21/S56/s3/T1'
# for instance in tqdm.tqdm(range(286)):
#     generate_npy(task_dir=task, instance=instance, camera_id=2, save_color_pcld=False, hand_pose=True)
# generate_npy('/mnt/data/hewang/h2o_data/HOI4D/C5/N50/S37/s2/T2', instance=60, camera_id=2, save_color_pcld=True, hand_pose=True)