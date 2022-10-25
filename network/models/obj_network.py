import sys
import os
from os.path import join as pjoin
from configs.config import get_config

from network.models.hand_utils import canonicalize
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backbones import PointNet2Msg_fast
from blocks import get_point_mlp, RotationRegressor
from pose_utils.part_dof_utils import merge_reenact_canon_part_pose, convert_pred_rtvec_to_matrix, eval_part_full, compute_parts_delta_pose
from pose_utils.procrustes import scale_pts_mask, translate_pts_mask, transform_pts_2d_mask, rot_around_yaxis_to_3d
from pose_utils.pose_fit import part_fit_st_no_ransac
from loss import compute_miou_loss, compute_nocs_loss, compute_part_dof_loss
from utils import cvt_torch
from optimization_obj import get_RT,CatCS2InsCS,InsCS2CatCS

EPS = 1e-6

class CoordNet(nn.Module):
    def __init__(self, cfg):
        super(CoordNet, self).__init__()
        self.backbone = PointNet2Msg_fast(cfg, cfg['network']['backbone_out_dim'],
                                     net_type='camera', use_xyz_feat=True)
        in_dim = cfg['network']['backbone_out_dim']
        self.num_parts = cfg['num_parts']
        self.sym = cfg['obj_sym']
        seg_dim = self.num_parts + 1        #TODO: shall we add background?
        self.seg_head = get_point_mlp(in_dim, seg_dim, [], acti='none', dropout=None)
        self.nocs_head = get_point_mlp(in_dim, 3 * self.num_parts,
                                       cfg['network']['nocs_head_dims'],
                                       acti='sigmoid', dropout=None)
        self.device = cfg['device']
        self.pose_loss_type = cfg['pose_loss_type']
        self.dataset_name = cfg['data_cfg']['dataset_name']
        self.ckpt_category = cfg['rot_exp']['dir'].split('_')[-1]

    def forward(self, input, flag_dict):
        #canonicalization
        cam = cvt_torch(input['points'], self.device).transpose(-1,-2)  # [B, 3, N]
        init_obj_poses = cvt_torch(input['jittered_obj_pose'], self.device)          #use pose of the first object part 
        canon_obj_poses = {key: init_obj_poses[key][:, 0, ...] for key in init_obj_poses.keys()}
        
        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB':
            cam = torch.matmul(canon_obj_poses['rotation'].transpose(-1, -2), cam - canon_obj_poses['translation'])
            normalization_pth = os.path.join('/data/h2o_data/HO3D/SDF/NormalizationParameters/%s/textured_simple.npz' % input['category'][0])
            normalization_params = np.load(normalization_pth)
            normalization_params = {
                'scale': normalization_params['scale'] / 2,
                'offset': normalization_params['offset']
            }
            cam = CatCS2InsCS(cam.transpose(-1,-2), normalization_params, input['category'][0]).transpose(-1,-2)
            if self.ckpt_category == 'car':
                change_axis = torch.zeros((3,3)).to(self.device)
                change_axis[0,1] = 1
                change_axis[1,2] = 1
                change_axis[2,0] = 1
                cam = torch.matmul(change_axis[None,:], cam)
        elif self.dataset_name == 'newShapeNet':
            cam = canonicalize(cam, canon_obj_poses)
        else: 
            print(self.dataset_name)
            np.savetxt('unknowndataset.txt', cam[0].transpose(-1,-2).cpu().numpy())
            raise NotImplementedError

        feat = self.backbone(cam)  # [B, backbone_out_dim, N]
        seg = self.seg_head(feat)  # [B, P+1, N]
        seg = F.softmax(seg, dim=1)
        nocs = self.nocs_head(feat) - 0.5   #[B, 3*P, N]
        

        if self.dataset_name == 'HO3D' or  self.dataset_name == 'DexYCB':
            if self.ckpt_category == 'car':
                change_axis = torch.zeros((3,3)).to(self.device)
                change_axis[0,1] = 1
                change_axis[1,2] = 1
                change_axis[2,0] = 1
                nocs = torch.matmul(change_axis[None,:].transpose(-1,-2), nocs)
            nocs = InsCS2CatCS(nocs.transpose(-1,-2), normalization_params, input['category'][0]).transpose(-1,-2)
        elif self.dataset_name == 'newShapeNet':
            pass 
        else: 
            raise NotImplementedError

        pred_labels = torch.argmax(seg, dim=-2)
        ret_dict = {
            'pred_seg': seg,            #[B, P+1, N]
            'pred_labels': pred_labels,  #[B, N]
            'pred_nocs': nocs.reshape(len(nocs), self.num_parts, 3, -1),          #[B, P, 3, N]
           # 'canon_obj_poses_coordnet': canon_obj_poses    
        }        
        return ret_dict

    def compute_loss(self, input, ret_dict, flag_dict):
        loss_dict = {}

        test_flag = flag_dict['test_flag']
        track_flag = flag_dict['track_flag']

        gt_labels = input['labels'].long().to(self.device)
        gt_nocs = cvt_torch(input['obj_nocs'], self.device).transpose(-1,-2)
        loss_dict['seg_loss'] = compute_miou_loss(ret_dict['pred_seg'], gt_labels, per_instance=False)
        labels = ret_dict['pred_labels'] if test_flag else gt_labels
        nocs_loss = compute_nocs_loss(ret_dict['pred_nocs'].reshape(len(ret_dict['pred_nocs']), self.num_parts*3, -1), gt_nocs, labels=labels,
                                      confidence=None, loss='l2', self_supervise=False,
                                      per_instance=False, sym=(self.sym == 1))
        if self.sym == 1:
            loss_dict['nocs_dist_loss'], loss_dict['nocs_pwm_loss'] = nocs_loss
            loss_dict['nocs_loss'] = 0
        else:
            loss_dict['nocs_dist_loss'], loss_dict['nocs_pwm_loss'] = 0,0
            loss_dict['nocs_loss'] = nocs_loss

        
        if not track_flag:          # use gt rotation to compute T, s. Don't do it while tracking
            gt_obj_poses = cvt_torch(input['gt_obj_pose'], self.device)
            init_obj_poses = cvt_torch(input['jittered_obj_pose'], self.device)
            rotation = gt_obj_poses['rotation']
            pred_obj_poses = {'rotation': rotation}
            pred_npcs = ret_dict['pred_nocs']     #[B, P, 3, N]
            cam_points = cvt_torch(input['points'], self.device).unsqueeze(1).repeat(1, self.num_parts, 1, 1).transpose(-1,-2)

            eye = torch.cat([torch.eye(self.num_parts), torch.zeros(2, self.num_parts)], dim=0).to(pred_npcs.device)
            mask = eye[labels, ].transpose(-1, -2)  # [B, N, P] --> [B, P, N]
            valid_mask = (mask.sum(dim=-1) > 0).float()
            
            if self.sym == 1:
                canon_cam = torch.matmul(rotation.transpose(-1, -2), cam_points)   # [B, P, 3, N]
                src_2d = pred_npcs[..., [0, 2], :].transpose(-1, -2)  # [B, P, N, 2]
                tgt_2d = canon_cam[..., [0, 2], :].transpose(-1, -2)
                rot_2d, _ = transform_pts_2d_mask(src_2d, tgt_2d, mask.unsqueeze(-1))
                rot_3d = rot_around_yaxis_to_3d(rot_2d)
                rotated_npcs = torch.matmul(rotation, torch.matmul(rot_3d, pred_npcs))
            else:
                rotated_npcs = torch.matmul(rotation, pred_npcs)

            scale_mask = mask.unsqueeze(-2)  # [B, P, 1, N]

            def center(source, mask):
                source_center = torch.sum(source * mask, dim=-1, keepdim=True) / torch.clamp(torch.sum(mask, dim=-1, keepdim=True), min=1.0)
                source_centered = (source - source_center.detach()) * mask  # [B, P, 3, N]
                return source_centered
            pred_obj_poses['scale'] = scale_pts_mask(center(rotated_npcs, scale_mask), center(cam_points, scale_mask), scale_mask)
            pred_obj_poses['scale'] = (valid_mask * pred_obj_poses['scale'] + (1.0 - valid_mask) * init_obj_poses['scale'])
            invalid_scale_mask = torch.logical_or(torch.isnan(pred_obj_poses['scale']), torch.isinf(pred_obj_poses['scale'])).float()
            pred_obj_poses['scale'] = (1.0 - invalid_scale_mask) * pred_obj_poses['scale'] + invalid_scale_mask * init_obj_poses['scale']

            scale = pred_obj_poses['scale'] if test_flag else gt_obj_poses['scale']

            scaled_npcs = scale.unsqueeze(-1).unsqueeze(-1) * rotated_npcs  # [B, P, 3, N]
            pred_obj_poses['translation'] = translate_pts_mask(scaled_npcs, cam_points, mask.unsqueeze(-1))

            pred_obj_poses['translation'] = (valid_mask.unsqueeze(-1).unsqueeze(-1) * pred_obj_poses['translation']
                                         + (1.0 - valid_mask.unsqueeze(-1).unsqueeze(-1)) * init_obj_poses['translation'])
            invalid_trans_mask = torch.logical_or(torch.isnan(pred_obj_poses['translation'].sum((-1, -2))),
                                                  torch.isinf(pred_obj_poses['translation'].sum((-1, -2)))).float().unsqueeze(-1).unsqueeze(-1)
            pred_obj_poses['translation'] = (1.0 - invalid_trans_mask) * pred_obj_poses['translation'] + invalid_trans_mask * init_obj_poses['translation']
           
            ret_dict['gt_obj_poses'] = gt_obj_poses
            ret_dict['pred_obj_poses'] = pred_obj_poses
            pose_diff, per_diff = eval_part_full(gt_obj_poses, pred_obj_poses, axis=int(self.sym))
            init_pose_diff, init_per_diff = eval_part_full(gt_obj_poses, init_obj_poses, axis=int(self.sym))
            loss_dict.update(pose_diff)
            loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})
            loss_dict.update(compute_part_dof_loss(gt_obj_poses, pred_obj_poses, self.pose_loss_type))
        return loss_dict, ret_dict
    
    def visualize(self, input, ret_dict, flag_dict):
        # visualize
        def group_pts(pts, labels):
            max_l = np.max(labels)
            pt_list = []
            for p in range(max_l + 1):
                idx = np.where(labels == p)
                pt_list.append(pts[idx])
            return pt_list
        cam = input['points']
        labels = ret_dict['labels']

        bb = 0
        s = []
        ss = []
        for i in range(3):
            ixd = np.where(input['labels'][bb].cpu().numpy() == i)
            s.append(input['obj_nocs'][bb].cpu().transpose(1, 0)[ixd])
            ixd = np.where(labels[bb].cpu().numpy() == i)
            ss.append(ret_dict['nocs'][bb].cpu().transpose(1, 0)[ixd])
        from vis_utils import plot3d_pts
        plot3d_pts(
            [group_pts(cam[bb].cpu().transpose(1, 0).numpy(), input['labels'][bb].cpu().numpy()),
                group_pts(cam[bb].cpu().transpose(1, 0).numpy(), labels[bb].cpu().numpy()),
                [s[0]], [s[1]], [ss[0]], [ss[1]]],
            show_fig=True)
        return          

class RotationRegressionBackbone(nn.Module):
    def __init__(self, cfg, rot_kind):
        super(RotationRegressionBackbone, self).__init__()
        self.num_parts = cfg['num_parts']
        self.encoder = PointNet2Msg_fast(cfg, cfg['network']['backbone_out_dim'], use_xyz_feat=False)
        self.sym = cfg['obj_sym']
        if rot_kind in ['6d', '3d']:        #So far 3d is only for sym obj
            self.rot_dim = int(rot_kind[0])
        else:
            raise NotImplementedError
        self.pose_pred = RotationRegressor(cfg['network']['backbone_out_dim'], self.num_parts, self.rot_dim)

    def forward(self, cam, cam_labels):  # [B*P, 3, N], [B*P, N]
        feat = self.encoder(cam)        #[B*P, 3, N] -> [B*P, dim, N]
        batch = int(cam.shape[0]/self.num_parts)
        eye_mat = torch.cat([torch.eye(self.num_parts), torch.zeros(1, self.num_parts)], dim=0)         #0<= label <= P+1
        part_mask = eye_mat[cam_labels,].to(cam_labels.device).transpose(-1, -2)         #[B*P, P, N]
        part_mask = part_mask.reshape(batch, self.num_parts, self.num_parts, -1)             #[B, P, P, N]
        part_mask = torch.stack([part_mask[:,i,i,:] for i in range(self.num_parts)], dim=1).unsqueeze(-2)      #[B, P, 1, N]
        valid_mask = (part_mask.sum(dim=-1) > 0).bool()                #[B, P, 1]
        raw_pred = self.pose_pred(feat)                          # [B, P, rot_dim, N]
        weighted_pred = (raw_pred * part_mask).sum(-1)  / torch.clamp(part_mask.sum(-1), min=1.0)  # [B, P, D] / [B, P, 1]

        if self.sym == 1:
            default = torch.tensor((0, 1, 0))
        else:
            default = torch.eye(3).reshape(-1)
            if self.rot_dim == 6:
                default = default[:6]
            else:
                raise NotImplementedError
        weighted_pred = valid_mask * weighted_pred  + (~valid_mask) * default.float().to(raw_pred.device).reshape(1, 1, -1)  #if one part is missing, use default to replace it.
        return weighted_pred


class RotationNet(nn.Module):
    def __init__(self, cfg):
        super(RotationNet, self).__init__()
        self.r_kind = cfg['pose_loss_type']['r'][-2:]
        self.regress_net = RotationRegressionBackbone(cfg, self.r_kind)
        self.device = cfg['device']
        self.num_parts = cfg['num_parts']
        self.sym = cfg['obj_sym'] 
        self.pose_loss_type = cfg['pose_loss_type']
        self.dataset_name = cfg['data_cfg']['dataset_name']
        self.ckpt_category = cfg['rot_exp']['dir'].split('_')[-1]
        print('use', self.r_kind, 'representation!')

    def forward(self, input, flag_dict):
        """
        If "eval_baseline rnpcs", pass pred_labels and pred_nocs by input
        """
        track_flag = flag_dict['track_flag']
        
        canon_obj_poses = cvt_torch(input['jittered_obj_pose'], self.device)
        canon_poses = {key: canon_obj_poses[key].reshape((-1,) + canon_obj_poses[key].shape[2:])  # [B, P, x] --> [B * P, x]
                      for key in ['rotation', 'translation', 'scale']}

        if track_flag:
            cam = cvt_torch(input['obj_points'], self.device)  # [B, N, 3]
        else:
            cam = cvt_torch(input['points'], self.device)  # [B, N, 3]
        batch_size = len(cam)
        cam_seg = input['pred_labels'] if track_flag else input['labels'].long().to(self.device)
        cam = cam.unsqueeze(1).repeat(1, self.num_parts, 1, 1).reshape((-1, ) + cam.shape[-2:]).transpose(-1,-2)  # [B*P, 3, N]
        cam_seg = cam_seg.unsqueeze(1).repeat(1, self.num_parts, 1).reshape((-1, ) + cam_seg.shape[-1:])  # [B * P, N]

        # canonicalize
        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB':
            canon_cam = torch.matmul(canon_poses['rotation'].transpose(-1, -2), cam - canon_poses['translation'])
            normalization_pth = os.path.join('/data/h2o_data/HO3D/SDF/NormalizationParameters/%s/textured_simple.npz' % input['category'][0])
            normalization_params = np.load(normalization_pth)
            normalization_params = {
                'scale': normalization_params['scale'] / 2,
                'offset': normalization_params['offset']
            }
            canon_cam = CatCS2InsCS(canon_cam.transpose(-1,-2), normalization_params, input['category'][0]).transpose(-1,-2)
            if self.ckpt_category == 'car':
                change_axis = torch.zeros((3,3)).to(self.device)
                change_axis[0,1] = 1
                change_axis[1,2] = 1
                change_axis[2,0] = 1
                canon_cam = torch.matmul(change_axis[None,:], canon_cam)
        elif self.dataset_name != 'newShapeNet': 
            canon_cam = canonicalize(cam, canon_poses)
        else:
            print(self.dataset_name)
            np.savetxt('unknowndataset.txt', cam[0].transpose(-1,-2).cpu().numpy())
            raise NotImplementedError

        # network
        raw_pred = self.regress_net(canon_cam, cam_seg)       #[B, num_parts, rot_dim]

        tmp = {}
        tmp['rotation'] = convert_pred_rtvec_to_matrix(raw_pred, self.sym == 1, self.r_kind)     #inference
        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB':
            R, _ = get_RT(input['category'][0])
            R = torch.FloatTensor(R).to(self.device)[None,...]
            if self.ckpt_category == 'car':
                R = torch.matmul(change_axis[None,:], R)
            tmp['rotation'] = torch.matmul(torch.matmul(R.transpose(-1,-2), tmp['rotation']), R)

        if not track_flag:
            gt_obj_poses = cvt_torch(input['gt_obj_pose'], self.device)
            pred_obj_poses = merge_reenact_canon_part_pose(canon_obj_poses, tmp)
            pred_obj_poses['translation'] = gt_obj_poses['translation']
            pred_obj_poses['scale'] = gt_obj_poses['scale']
        else:
            merged_pose = merge_reenact_canon_part_pose(canon_obj_poses, tmp)
            rotation = merged_pose['rotation']
            pred_obj_poses = {'rotation': rotation}
            labels = input['pred_labels']
            pred_npcs = input['pred_nocs']      # [B, P, 3, N]
            gt_obj_poses = cvt_torch(input['gt_obj_pose'], self.device)
            if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB':
                scale = torch.ones_like(gt_obj_poses['scale']).to(self.device)
            else:
                scale = gt_obj_poses['scale']
            pred_obj_poses, valid = part_fit_st_no_ransac(labels, pred_npcs.transpose(-1, -2), cam.reshape(batch_size, self.num_parts, 3, -1).transpose(-1,-2),
                                               rotation, {'num_parts': self.num_parts, 'sym': self.sym == 1}, given_scale=scale)
            pred_obj_poses['scale'] = valid.float() * gt_obj_poses['scale'] + (1.0 - valid.float()) * canon_obj_poses['scale']
            # pred_obj_poses['scale'] = valid.float() * pred_obj_poses['scale'] + (1.0 - valid.float()) * canon_obj_poses['scale']
            pred_obj_poses['translation'] = (valid.float().unsqueeze(-1).unsqueeze(-1) * pred_obj_poses['translation']
                                         + (1.0 - valid.float().unsqueeze(-1).unsqueeze(-1)) * canon_obj_poses['translation'])
            
        ret_dict = {'canon_obj_poses_rotnet': canon_poses,  #[B*P, x]
                    'pred_obj_poses': pred_obj_poses,   #[B, P, x]
                    'raw_rotation': raw_pred}           #[B, P, x]
        return ret_dict

    def compute_loss(self, input, ret_dict, flag_dict):
        loss_dict = {}
        init_obj_poses = cvt_torch(input['jittered_obj_pose'], self.device)
        gt_obj_poses = cvt_torch(input['gt_obj_pose'], self.device)
        pred_obj_poses = ret_dict['pred_obj_poses']
        # gt_corners = feed_dict['meta']['obj_nocs_corners'].float().to(self.device)
        # if self.sym:
        #     gt_bbox = yaxis_from_corners(gt_corners, self.device)
        # else:
        #     gt_bbox = tensor_bbox_from_corners(gt_corners, self.device)
        # corner_loss, corner_per_diff = compute_point_pose_loss(feed_dict['gt_part'], pred_dict['part'],
        #                                                         gt_bbox,
        #                                                         metric=self.pose_loss_type['point'])
        # loss_dict['corner_loss'] = corner_loss
        pose_diff, per_diff = eval_part_full(gt_obj_poses, pred_obj_poses,
                                             per_instance=False, axis=int(self.sym))
        init_pose_diff, init_per_diff = eval_part_full(gt_obj_poses, init_obj_poses,
                                                       per_instance=False,
                                                       axis=int(self.sym))
        loss_dict.update(pose_diff)
        loss_dict.update({f'init_{key}': value for key, value in init_pose_diff.items()})

        batch_size = len(gt_obj_poses['scale']) 
        gt_delta_poses = compute_parts_delta_pose(init_obj_poses, gt_obj_poses,init_obj_poses)
        gt_delta_rotation = gt_delta_poses['rotation']  #[B, P, x]
        #current implementation of rot loss: l1(mean(raw_xi), gt)
        if self.sym == 1:  
            if self.pose_loss_type['r'] == 'raw_3d':  
                rloss = (gt_delta_rotation[..., 1] - ret_dict['raw_rotation']).abs().mean(dim=-1)
            else:
                raise NotImplementedError
                rloss = rot_yaxis_loss(gt_delta_rotation, gt_delta_rotation)  # todo: need to change
        else:
            if self.pose_loss_type['r'] == 'raw_6d':
                rloss = ((gt_delta_rotation[..., 0] - ret_dict['raw_rotation'][..., 0:3]).abs() + (
                            gt_delta_rotation[..., 1] - ret_dict['raw_rotation'][..., 3:6]).abs()).mean(dim=-1)
            elif self.pose_loss_type['r'] == 'raw_9d':
                tmp = ret_dict['raw_rotation'].reshape(ret_dict['raw_rotation'].shape[:-1] + (3, 3))
                rloss = (gt_delta_rotation - tmp).abs().reshape(tmp.shape[:-2] + (-1,)).mean(dim=-1)
            else:
                raise NotImplementedError
                rloss = rot_trace_loss(gt_delta_rotation, gt_delta_rotation,
                                        metric=self.pose_loss_type['r'])  # todo: need to change
        loss_dict['rloss'] = rloss.mean()
        ret_dict['gt_rotation'] = gt_delta_rotation
        return loss_dict, ret_dict
        
    def visualize(self, input, ret_dict, flag_dict):
        rand_num = np.random.randint(0, ret_dict['raw_rotation'].shape[0])
        pc = input['points'].to(self.device)
        canon_pose = ret_dict['canon_obj_poses_rotnet']
        lst = []
        for i in range(100):
            lst.append(ret_dict['gt_rotation'][...,1] / 100*i)
        lst = torch.stack(lst, dim=1).squeeze()
        
        from vis_utils import plot3d_pts
        print(canon_pose['rotation'].shape, pc.shape)
        pc_objframe = canonicalize(pc.transpose(-1,-2), canon_pose)
        print(lst.shape)
        plot3d_pts(
            [[pc_objframe.transpose(-1,-2).cpu()[rand_num], lst.cpu()[rand_num]]],
            show_fig=True)
        return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='2.25_segmentation_HO3D_0.001_CE_RGB_small_norot.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args)
   