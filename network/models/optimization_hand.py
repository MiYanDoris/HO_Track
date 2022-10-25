import numpy as np
import torch
import pickle
from our_mano import OurManoLayer
from network.models.hand_utils import mano_axisang2quat, mano_quat2axisang
from pose_utils.rotations import matrix_to_unit_quaternion, unit_quaternion_to_matrix
from DeepSDF import Decoder
import os
import cv2
from optimization_obj import CatCS2InsCS, InsCS2CatCS
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)


def world2mask(xyz, device):
    '''
        xyz: B, N, 3
    '''
    B = xyz.shape[0]
    mask = torch.zeros((B, h, w), device=device, dtype=torch.bool)
    x = (xyz[..., 0] / xyz[..., 2] * fx + w // 2).long()
    y = (xyz[..., 1] / xyz[..., 2] * fy + h // 2).long()
    x = torch.clamp(x, 0, w-1)
    y = torch.clamp(y, 0, h-1)
    help = torch.arange(x.shape[0])[:,None].repeat(1, x.shape[1])
    mask[help, y, x] = True
    return mask.bool()

def world2point2D(xyz, fx, fy, cx, cy):
    '''
        xyz: B, N, 3
    '''
    B, N, _ = xyz.shape
    x = (xyz[..., 0] / xyz[..., 2] * fx + cx).reshape(B, N, 1)
    y = (xyz[..., 1] / xyz[..., 2] * fy + cy).reshape(B, N, 1)
    point_2D = torch.cat([y, x], dim=-1).float()
    return point_2D # B, N, 2


def kp2length(kp):
    bone_mask = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
    parent_mask = [0,1,2,0,5,6,0,9,10,0,13,14,0,17,18]
    bone_length =  torch.norm(kp[:, bone_mask] - kp[:, parent_mask], dim=-1) # B, 15
    return bone_length

# To solve MANO shape code from hand joints
class optimize_hand_beta():
    def __init__(self, cfg):
        #parameters
        self.particle_size = 5120
        self.iteration = 20
        self.optimize_dim = 10
        self.beta = 0.9
        self.device = cfg['device']
        self.loss_weight = cfg['loss_weight']
        print('Need to optimize %d dims' % self.optimize_dim)

        self.initial_scale = torch.ones(self.optimize_dim, device=self.device) * 5
        self.scaling_coefficient2 = 2000      

        # MANO
        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right').cuda()
        
        # sample particles
        mean = np.zeros(self.optimize_dim)
        cov = np.eye(self.optimize_dim) 
        self.pre_sampled_particle = np.random.multivariate_normal(mean, cov, self.particle_size)
        self.pre_sampled_particle[0,:] = 0
        self.pre_sampled_particle = torch.FloatTensor(self.pre_sampled_particle).to(self.device)  

    def evaluate(self, kp):   
        energy = (kp2length(kp).unsqueeze(1) - self.old_pred_length).abs().mean(dim=-1).mean(dim=-1) 
        # print('energy for current solution:', energy[0])
        # print('sampled energy mean:', energy.mean())
        # print('sampled energy min:', energy.min())
        # print('sampled energy max:', energy.max())
        # print('---------------------------')
        return energy 

    def update_seach_size(self, energy, mean_transform):
        s = mean_transform.abs() + 1e-3
        search_size = energy * self.scaling_coefficient2 * s / s.norm() + 1e-3
        return search_size

    def set_init_para(self, pred_kp, use_old):
        self.hand_shape = torch.zeros((1,10)).to(self.device)
        self.pred_length = kp2length(pred_kp).unsqueeze(1)
        if use_old:
            try:
                self.old_pred_length = torch.cat([self.old_pred_length, self.pred_length], dim=1)
            except:
                self.old_pred_length = self.pred_length
        else:
            self.old_pred_length = self.pred_length
            
    def optimize(self, pred_kp, use_old=False):
        # initialize 
        if pred_kp.shape[-2] == 21:
            self.set_init_para(pred_kp, use_old)
        else:
            print(pred_kp.shape)
            raise NotImplementedError

        search_size = self.initial_scale
        prev_search_size = search_size
        count = 0
        prev_success_flag = True

        while True:
            if count == self.iteration:
                break 

            # get new sample
            sample = self.pre_sampled_particle*search_size
            _, kp = self.mano_layer_right.forward(th_pose_coeffs=torch.zeros((self.particle_size,48)).to(self.device),
                                                 th_trans=torch.zeros((self.particle_size,3)).to(self.device),
                                                 th_betas=self.hand_shape+sample)
            energy = self.evaluate(kp)    # [B]

            origin_energy = energy[0]
            better_mask = energy < origin_energy  #[B]
            weight = (origin_energy - energy) * better_mask      #[B]
            weight_sum = weight.sum() 
            if torch.any(better_mask):
                mean_energy = (energy * weight).sum() / weight.sum()
                success_flag = True  
            else:
                mean_energy = energy[0]  
                success_flag = False 

            #update R, T
            if success_flag:
                mean_transform = (sample * weight.unsqueeze(1)).sum(dim=0, keepdim=True) / weight_sum    #[1, 10]
                self.hand_shape += mean_transform
            else:
                mean_transform = torch.zeros((1,10), device=self.device)

            search_size = self.update_seach_size(mean_energy, mean_transform)
            if prev_success_flag and success_flag:
                search_size = self.beta * search_size + (1-self.beta)*prev_search_size
                prev_search_size = search_size
            elif success_flag:
                prev_search_size = search_size
            prev_success_flag = success_flag
            count += 1
        return self.hand_shape


class optimize_hand_joint():
    '''
    Optimize hand joint pose to improve the HO interation.
    We use the similar gradient-free optimization method as in RoseFusion[Siggraph2021]
    It is quite tricky to tune the hyper-parameter if you want to improve the kp error and physical interaction at the same time.
    '''
    def __init__(self, cfg):
        #parameters
        self.particle_size = cfg['network']['particle_size']
        self.iteration = cfg['network']['iterations']
        self.root_dir = cfg['data_cfg']['basepath']
        self.theta_scale = 30  
        self.ncomps = 10
        self.optimize_dim = 6+self.ncomps
        self.beta = 0.9
        self.device = cfg['device']
        self.loss_weight = cfg['loss_weight']
        print('Need to optimize %d dims' % self.optimize_dim)

        self.initial_scale = torch.ones(self.optimize_dim, device=self.device) * 0.005 
        self.scaling_coefficient2 = 0.1
        self.volume_size = 151
        self.voxel_scale = 0.003

        # MANO
        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right', add_points=cfg['add_points']).cuda()
        
        # sample particles
        mean = np.zeros(self.optimize_dim)
        cov = np.eye(self.optimize_dim) 
        self.pre_sampled_particle = np.random.multivariate_normal(mean, cov, self.particle_size)
        self.pre_sampled_particle[0,:] = 0
        self.pre_sampled_particle = torch.FloatTensor(self.pre_sampled_particle).to(self.device)  
        self.data_config = cfg['data_config']

        # contact region
        # NOTE: These contact zones comes from "Obman"
        with open(pjoin(BASEPATH, 'contact_zones.pkl'), "rb") as p_f:
            contact_data = pickle.load(p_f)
        self.hand_region = contact_data["contact_zones"] # zone_idx, zone_vert_idxs
        self.tips_region = []
        self.finger_mask = []
        for i in range(5):
            prev_len = len(self.tips_region)
            self.tips_region.extend(self.hand_region[i+1])
            self.finger_mask.append(list(range(prev_len, len(self.tips_region))))

        # DeepSDF for object
        latent_size = 256
        self.SDFDecoder = Decoder(latent_size, **cfg['network']["NetworkSpecs"])
        self.SDFDecoder = torch.nn.DataParallel(self.SDFDecoder)
        
        # pre-defined variables for voxelization
        self.volume_ind = torch.arange(self.volume_size**3)[:,None].repeat(1,3)
        self.volume_ind[:, 2] = self.volume_ind[:, 2] % self.volume_size
        self.volume_ind[:, 1] = self.volume_ind[:, 1] // self.volume_size % self.volume_size
        self.volume_ind[:, 0] = self.volume_ind[:, 0] // self.volume_size // self.volume_size
        self.volume_ind = (self.volume_ind - self.volume_size//2) * self.voxel_scale
        self.volume_ind = self.volume_ind.to(self.device)
        
        self.dataset_name = cfg['data_cfg']['dataset_name']
        
    def load_obj(self, obj_info, instance):
        latent_code_pth, self.normalization_param, saved_model_pth = obj_info 
        saved_model_state = torch.load(saved_model_pth)
        self.SDFDecoder.load_state_dict(saved_model_state["model_state_dict"])
        SDFDecoder = self.SDFDecoder.module.to(self.device)

        print('load sdf code from %s' % latent_code_pth)
        self.latent_code = torch.load(latent_code_pth)[0][0].to(self.device) # 1, 1, L

        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB': # bottle, can, box, bowl
            ins_volume_ind = CatCS2InsCS(self.volume_ind, self.normalization_param, instance)
        elif self.dataset_name == 'newShapeNet':
            ins_volume_ind = (self.volume_ind + torch.FloatTensor(self.normalization_param['offset']).to(self.device))*torch.FloatTensor(self.normalization_param['scale']).to(self.device)
        else:
            raise NotImplementedError

        latent_inputs = self.latent_code.expand(ins_volume_ind.shape[0], -1)
        inputs = torch.cat([latent_inputs, ins_volume_ind], 1)
        self.sdf_volume = SDFDecoder(inputs).reshape(self.volume_size,self.volume_size,self.volume_size) / self.normalization_param['scale'][0]        #[V^3, 1]
        return 

    def get_kp_from_delta(self, delta):
        '''
            normalized_delta: B, optimize_dim
        '''
        sampled_r = torch.matmul(self.curr_r, unit_quaternion_to_matrix(delta[:, :4])) 
        sampled_t = self.curr_t + delta[:, 4:7, None]
        sampled_theta = self.curr_theta + self.mano_layer_right.pca_comps2pose(self.ncomps, delta[:, 7:])*self.theta_scale
        sampled_axisangle = mano_quat2axisang(matrix_to_unit_quaternion(sampled_r))
        hand, kp = self.mano_layer_right.forward(th_pose_coeffs=torch.cat([sampled_axisangle,sampled_theta], dim=-1),
                                         th_trans=sampled_t.squeeze(-1), use_registed_beta=True)
        return hand, kp 

    def get_regularization_loss(self, kp):
        error = (kp-self.pred_kp).norm(dim=-1)
        vis_regu = torch.sum(error*self.vis_mask, dim=-1) / torch.clamp(torch.sum(self.vis_mask, dim=-1), 1)
        invis_regu = torch.sum(error*(~self.vis_mask), dim=-1) / torch.clamp(torch.sum(~self.vis_mask, dim=-1), 1)  # B
        return vis_regu, invis_regu

    def get_silhouette_loss(self, hand):
        pred_2D = world2point2D(hand, self.proj['fx'][0], self.proj['fy'][0], self.proj['cx'][0], self.proj['cy'][0])   #[B, N, 2]
        index1 = torch.clamp(pred_2D[...,0].long(), 0, self.h-1)
        index2 = torch.clamp(pred_2D[...,1].long(), 0, self.w-1)
        silhouette_loss = self.gt_mask[index1, index2]
        silhouette_loss = silhouette_loss.sum(dim=-1) / pred_2D.shape[1]
        return silhouette_loss

    def get_attraction_loss(self, queried_sdf, threshold=0):
        assert self.vis_mask.shape[-1] == 21 and self.vis_mask.shape[0] == 1
        index = [8, 12, 16, 20, 4]
        invis_finger = ~self.vis_mask[0, index]
        tips_region_sdf = queried_sdf[:, self.tips_region] # B, 96
        tips_region_dis = tips_region_sdf * (tips_region_sdf > threshold)
        attr_loss_lst = [torch.min(tips_region_dis[:, self.finger_mask[i]], dim=-1)[0] for i in range(5) if invis_finger[i]]
        attraction_loss = sum(attr_loss_lst)
        return attraction_loss

    def query_sdf(self, hand):
        B, N, _ = hand.shape 
        pcld_flat = torch.matmul((hand - self.obj_t), self.obj_r).reshape(-1, 3) # B*N, 3
        pc_vol_indx = torch.clamp(pcld_flat[:,0]//self.voxel_scale, -(self.volume_size//2), self.volume_size//2).long() + self.volume_size//2
        pc_vol_indy = torch.clamp(pcld_flat[:,1]//self.voxel_scale, -(self.volume_size//2), self.volume_size//2).long() + self.volume_size//2
        pc_vol_indz = torch.clamp(pcld_flat[:,2]//self.voxel_scale, -(self.volume_size//2), self.volume_size//2).long() + self.volume_size//2
        assert pc_vol_indx.min() >= 0 and pc_vol_indx.max() < self.volume_size, "x: %d %d"%(pc_vol_indx.min(), pc_vol_indx.max())
        assert pc_vol_indy.min() >= 0 and pc_vol_indy.max() < self.volume_size, "y: %d %d"%(pc_vol_indy.min(), pc_vol_indy.max())
        assert pc_vol_indz.min() >= 0 and pc_vol_indz.max() < self.volume_size, "z: %d %d"%(pc_vol_indz.min(), pc_vol_indz.max())
        queried_sdf = self.sdf_volume[pc_vol_indx, pc_vol_indy, pc_vol_indz].reshape(B, N)
        return queried_sdf

    def get_penetration_loss(self, queried_sdf, threshold=0):
        '''
            input:
                hand:       B, 778, 3 in camera frame
        '''
        abs_distance = queried_sdf.abs()
        penetrate_mask = (queried_sdf < - threshold).bool()
        penetrate_max = torch.max(abs_distance * penetrate_mask, dim=-1)[0]
        return penetrate_max

    def get_temporal_smooth_loss(self, kp):
        if self.last_frame_kp is None:
            smooth_loss = 0
        else:
            smooth_loss = torch.norm(kp-self.last_frame_kp, dim=-1).mean(dim=1)
        return smooth_loss

    def evaluate(self, hand, kp):   #B=particle_size N=point_size
        queried_sdf = self.query_sdf(hand)
        loss_dict = {}
        loss_dict['sil_loss'] = self.get_silhouette_loss(hand)
        loss_dict['penetrate_sum_loss'] = self.get_penetration_loss(queried_sdf)
        loss_dict['vis_regu_loss'], loss_dict['invis_regu_loss'] = self.get_regularization_loss(kp)
        loss_dict['temporal_smooth'] = self.get_temporal_smooth_loss(kp)
        if loss_dict['penetrate_sum_loss'][0] != 0:
            loss_dict['attraction_loss'] = self.get_attraction_loss(queried_sdf)
        else:
            loss_dict['attraction_loss'] = 0

        energy = 0
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] * self.loss_weight[key]
            energy += loss_dict[key]
        return energy 

    def update_seach_size(self, energy, mean_transform):
        s = mean_transform.abs() + 1e-3
        search_size = energy * self.scaling_coefficient2 * s / s.norm() + 1e-3
        return search_size
    
    def set_init_para(self, init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection):
        hand_shape = torch.FloatTensor(hand_shape.cpu()).to(self.device).reshape(1, 10)
        self.mano_layer_right.register_beta(hand_shape)
        self.pred_kp = init_kp
        self.last_frame_kp = last_frame_kp
        self.vis_mask = vis_mask
        self.proj = projection
        self.w = projection['w'][0]
        self.h = projection['h'][0]
        self.curr_t = (init_hand_pose['translation'].reshape(1, 3, 1)).to(self.device)
        self.curr_r = (init_hand_pose['rotation']).to(self.device)
        self.curr_theta = init_mano
        self.obj_r = init_obj_pose['rotation'].to(self.device).reshape(3, 3).float()
        self.obj_t = init_obj_pose['translation'].to(self.device).reshape(1, 1, 3).float()
        if self.data_config == 'data_info_HO3D.yml':
            silhouette_pth = pjoin(self.root_dir, 'pred_mask/ensemble/%s/%s.png' % (file_name.split('/')[0], file_name.split('/')[1]))
            mask = (cv2.imread(silhouette_pth) / 70)[:240] # 0:obj,1:hand,2back
            mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)[:, :, 0]
            self.gt_mask = (mask == 2)

        elif self.data_config == 'data_info_newShapeNet.yml':
            silhouette_pth = pjoin(self.root_dir, 'img/%s/seq/%s/mask.png' % (category, file_name))
            maskimg = cv2.imread(silhouette_pth)
            self.gt_mask = maskimg.sum(axis=-1) == 0
            
        self.gt_mask = torch.tensor(self.gt_mask).to(self.device)

        # elif self.data_config == 'data_info_HOI4D.yml':
        #     seq, id = file_name.split('/')[0], int(file_name.split('/')[1])
        #     mask_path = '/data/h2o_data/HOI4D/final/%s/%s/pred_mask/%06d.png' % (category, seq, id)
        #     mask = (cv2.imread(mask_path)[360:]) / 70
        #     mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)[:, :, 0]

        #     palm_mask_path = '/data/h2o_data/HOI4D/final/%s/%s/2Dseg/palm_mask/%05d.png' % (category, seq, id)
        #     obj_mask = (mask == 0)
        #     hand_mask = (mask == 1)

        #     palm_mask = cv2.imread(palm_mask_path)[:, :, 1] != 0
        #     hand_mask &= palm_mask

        #     self.gt_mask = torch.ones((1080, 1920)).to(self.device).float()
        #     self.gt_mask[obj_mask] = 0
        #     self.gt_mask[hand_mask] = 0

    def optimize(self, init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection):
        # initialize 
        self.set_init_para(init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection)
        search_size = self.initial_scale
        prev_search_size = search_size
        count = 0
        prev_success_flag = True

        while True:
            if count == self.iteration:
                break 

            # get new samples
            sample_part = self.pre_sampled_particle*search_size
            sample_qw = torch.sqrt(1-sample_part[:,0]**2-sample_part[:,1]**2-sample_part[:,2]**2).unsqueeze(1)
            sample = torch.cat([sample_qw, sample_part],dim=1)
            hand, kp = self.get_kp_from_delta(sample)

            energy = self.evaluate(hand, kp)    # [B]

            origin_energy = energy[0]
            better_mask = energy < origin_energy  #[B]
            weight = (origin_energy - energy) * better_mask      #[B]
            weight_sum = weight.sum() 
            if torch.any(better_mask):
                mean_energy = (energy * weight).sum() / weight.sum()
                success_flag = True  
            else:
                mean_energy = energy[0]  
                success_flag = False 

            #update R, T
            if success_flag:
                mean_transform = (sample * weight.unsqueeze(1)).sum(dim=0, keepdim=True) / weight_sum    #[1, 7]
                mean_transform[:, :4] /= mean_transform[:,:4].norm()
                self.curr_r = torch.matmul(self.curr_r, unit_quaternion_to_matrix(mean_transform[:, :4])) 
                self.curr_t = self.curr_t + mean_transform[:, 4:7, None]
                self.curr_theta = self.curr_theta + self.mano_layer_right.pca_comps2pose(self.ncomps, mean_transform[:, 7:])*self.theta_scale
            else:
                mean_transform = torch.zeros((1,7+self.ncomps), device=self.device)

            search_size = self.update_seach_size(mean_energy, mean_transform[:,1:])
            if prev_success_flag and success_flag:
                search_size = self.beta * search_size + (1-self.beta)*prev_search_size
                prev_search_size = search_size
            elif success_flag:
                prev_search_size = search_size
            prev_success_flag = success_flag
            count += 1
            # print(time.time()-s0)
        curr_axisangle = mano_quat2axisang(matrix_to_unit_quaternion(self.curr_r))
        final_hand, final_kp = self.mano_layer_right.forward(th_pose_coeffs=torch.cat([curr_axisangle,self.curr_theta], dim=-1), th_trans=self.curr_t.squeeze(-1), use_registed_beta=True)
       
        return final_kp, self.curr_theta, self.curr_r.squeeze(0), self.curr_t.squeeze(-1)



class adam_hand():
    def __init__(self, cfg):
        #parameters
        self.iteration = 100
        self.lr = 1e-3
        self.root_dir = cfg['data_cfg']['basepath']
        self.device = cfg['device']
        self.loss_weight = cfg['loss_weight']
        self.data_config = cfg['data_config']
        self.dataset_name = cfg['data_cfg']['dataset_name']
        print('Need to optimize %d dims' % self.optimize_dim)

        # MANO
        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right', add_points=cfg['add_points']).to(self.device)

        # contact region
        with open(pjoin(BASEPATH, 'contact_zones.pkl'), "rb") as p_f:
            contact_data = pickle.load(p_f)
        self.hand_region = contact_data["contact_zones"] # zone_idx, zone_vert_idxs
        self.tips_region = []
        self.finger_mask = []
        for i in range(5):
            prev_len = len(self.tips_region)
            self.tips_region.extend(self.hand_region[i+1])
            self.finger_mask.append(list(range(prev_len, len(self.tips_region))))

        # DeepSDF for object
        latent_size = 256
        self.SDFDecoder = Decoder(latent_size, **cfg['network']["NetworkSpecs"])
        self.SDFDecoder = torch.nn.DataParallel(self.SDFDecoder)

    def load_obj(self, obj_info, instance):
        latent_code_pth, self.normalization_param, saved_model_pth = obj_info 
        saved_model_state = torch.load(saved_model_pth)
        self.SDFDecoder.load_state_dict(saved_model_state["model_state_dict"])
        self.SDFDecoderm = self.SDFDecoder.module.to(self.device)

        print('load sdf code from %s' % latent_code_pth)
        self.latent_code = torch.load(latent_code_pth)[0][0].to(self.device) # 1, 1, L
        self.instance = instance
        return 

    def get_regularization_loss(self, kp):
        error = (kp-self.pred_kp).norm(dim=-1)
        vis_regu = torch.sum(error*self.vis_mask, dim=-1) / torch.clamp(torch.sum(self.vis_mask, dim=-1), 1)
        invis_regu = torch.sum(error*(~self.vis_mask), dim=-1) / torch.clamp(torch.sum(~self.vis_mask, dim=-1), 1)  # B
        return vis_regu.squeeze(), invis_regu.squeeze()

    def get_silhouette_loss(self, hand):
        pred_2D = world2point2D(hand, self.proj['fx'][0], self.proj['fy'][0], self.proj['cx'][0], self.proj['cy'][0])   #[B, N, 2]
        silhouette_dis,_ = (pred_2D- self.gt_mask_ind.unsqueeze(1)).norm(dim=-1).min(dim=0)
        silhouette_loss = silhouette_dis.mean(dim=-1)
        return silhouette_loss

    def get_attraction_loss(self, queried_sdf, threshold=0):
        assert self.vis_mask.shape[-1] == 21 and self.vis_mask.shape[0] == 1
        index = [8, 12, 16, 20, 4]
        invis_finger = ~self.vis_mask[0, index]
        tips_region_sdf = queried_sdf[self.tips_region] # B, 96
        tips_region_dis = tips_region_sdf * (tips_region_sdf > threshold)
        attr_loss_lst = [torch.min(tips_region_dis[self.finger_mask[i]], dim=-1)[0] for i in range(5) if invis_finger[i]]
        attraction_loss = sum(attr_loss_lst) 
        return attraction_loss

    def query_sdf(self, hand): 
        pcld_flat = torch.matmul((hand - self.obj_t), self.obj_r).reshape(-1, 3) # B*N, 3
        # TODO: coordinate transform.
        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB': # bottle, can, box, bowl
            pcld_flat = CatCS2InsCS(pcld_flat, self.normalization_param, self.instance)
        elif self.dataset_name == 'newShapeNet':
            pcld_flat = (pcld_flat + torch.FloatTensor(self.normalization_param['offset']).to(self.device))*torch.FloatTensor(self.normalization_param['scale']).to(self.device)
        else:
            raise NotImplementedError
        latent_inputs = self.latent_code.expand(pcld_flat.shape[0], -1)
        inputs = torch.cat([latent_inputs, pcld_flat], 1)
        queried_sdf = self.SDFDecoderm(inputs) / self.normalization_param['scale'][0]       #[V^3, 1]
        return queried_sdf.squeeze()

    def get_penetration_loss(self, queried_sdf, threshold=0):
        '''
            input:
                hand:       B, 778, 3 in camera frame
        '''
        abs_distance = queried_sdf.abs()
        penetrate_mask = (queried_sdf < - threshold).bool()
        penetrate_max = torch.sum(abs_distance * penetrate_mask, dim=-1)
        return penetrate_max

    def get_temporal_smooth_loss(self, kp):
        if self.last_frame_kp is None:
            smooth_loss = 0
        else:
            smooth_loss = torch.norm(kp-self.last_frame_kp, dim=-1).mean(dim=1).squeeze()
        return smooth_loss

    def evaluate(self, hand, kp):   #B=particle_size N=point_size
        queried_sdf = self.query_sdf(hand)
        loss_dict = {}
        loss_dict['sil_loss'] = self.get_silhouette_loss(hand)
        loss_dict['penetrate_sum_loss'] = self.get_penetration_loss(queried_sdf)
        loss_dict['vis_regu_loss'], loss_dict['invis_regu_loss'] = self.get_regularization_loss(kp)
        loss_dict['temporal_smooth'] = self.get_temporal_smooth_loss(kp)
        if loss_dict['penetrate_sum_loss'] != 0:
            loss_dict['attraction_loss'] = self.get_attraction_loss(queried_sdf)
        else:
            loss_dict['attraction_loss'] = 0

        energy = 0
        for key in loss_dict.keys():
            loss_dict[key] = loss_dict[key] * self.loss_weight[key]
            energy += loss_dict[key]
        return energy 

    def set_init_para(self, init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection):
        hand_shape = torch.FloatTensor(hand_shape.cpu()).to(self.device).reshape(1, 10)
        self.mano_layer_right.register_beta(hand_shape)
        self.pred_kp = init_kp
        self.last_frame_kp = last_frame_kp
        self.vis_mask = vis_mask
        self.proj = projection
        self.w = projection['w'][0]
        self.h = projection['h'][0]

        curr_axisangle = mano_quat2axisang(matrix_to_unit_quaternion((init_hand_pose['rotation']).to(self.device)))
        self.curr_coeff = torch.cat([curr_axisangle, init_mano], dim=-1)
        self.curr_coeff = torch.tensor(self.curr_coeff, device=self.device, requires_grad=True)
        self.curr_t = (init_hand_pose['translation'].reshape(1, 3)).to(self.device)
        self.curr_t = torch.tensor(self.curr_t, device=self.device, requires_grad=True)
        
        self.obj_r = init_obj_pose['rotation'].to(self.device).reshape(3, 3).float()
        self.obj_t = init_obj_pose['translation'].to(self.device).reshape(1, 1, 3).float()
        if self.data_config == 'data_info_HO3D.yml':
            # silhouette_pth = '/data/h2o_data/HO3D/mask/%s/aligned_mask/%s.png' % (file_name.split('/')[0], file_name.split('/')[1])
            # maskimg = cv2.imread(silhouette_pth)
            # self.gt_mask = maskimg.sum(axis=-1) == 0
            # self.gt_mask = torch.tensor(self.gt_mask).to(self.device)
            silhouette_pth = pjoin(self.root_dir, 'pred_mask/ensemble/%s/%s.png' % (file_name.split('/')[0], file_name.split('/')[1]))
            mask = (cv2.imread(silhouette_pth) / 70)[:240] # 0:obj,1:hand,2back
            mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)[:, :, 0]
            self.gt_mask = (mask == 2)

        elif self.data_config == 'data_info_newShapeNet.yml':
            silhouette_pth = pjoin(self.root_dir, 'img/%s/seq/%s/mask.png' % (category, file_name))
            maskimg = cv2.imread(silhouette_pth)
            self.gt_mask = maskimg.sum(axis=-1) == 0
            
        elif self.data_config == 'data_info_DexYCB.yml':
            # self.save_folder = file_name.split('+')[0]+'+'+file_name.split('+')[1]+'+'+file_name.split('+')[2]
            # self.save_name = file_name.split('+')[-1]
            silhouette_pth = pjoin(self.root_dir, 'tarfolder/%s/%s/%s/labels_%s.npz' % (file_name.split('+')[0],file_name.split('+')[1],file_name.split('+')[2],file_name.split('+')[3]))
            color_pth = silhouette_pth.replace('labels', 'color').replace('npz', 'jpg')
            rgbimg = cv2.imread(color_pth)
            maskimg = np.load(silhouette_pth)['seg']
            self.gt_mask = maskimg==0
            self.rgbimg = rgbimg * self.gt_mask[:,:,None]
        self.gt_mask = torch.tensor(self.gt_mask).to(self.device)
        # padded_mask = torch.nn.ZeroPad2d(padding=(1,1,1,1))(self.gt_mask)
        # boundary = (self.gt_mask==0) & (padded_mask[2:,1:-1] + padded_mask[:-2,1:-1] + padded_mask[1:-1,2:] + padded_mask[1:-1,:-2] != 0)
        gt_mask_x, gt_mask_y = torch.where(self.gt_mask==0)
        self.gt_mask_ind = torch.stack([gt_mask_x, gt_mask_y], dim=-1)

    def optimize(self, init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection, gttest=False):
        # initialize 
        self.set_init_para(init_mano, init_hand_pose, init_kp, last_frame_kp, vis_mask, init_obj_pose, category, file_name, hand_shape, projection)
        optimizer =  torch.optim.Adam([self.curr_coeff, self.curr_t], lr=self.lr)
        for i in range(self.iteration):
            optimizer.zero_grad()
            hand, kp = self.mano_layer_right.forward(th_pose_coeffs=self.curr_coeff,
                                         th_trans=self.curr_t, use_registed_beta=True)
            loss = self.evaluate(hand, kp)    # [B]
            loss.backward()
            optimizer.step()
        curr_coeff = self.curr_coeff.detach()
        curr_t = self.curr_t.detach()
        final_hand, final_kp = self.mano_layer_right.forward(th_pose_coeffs=curr_coeff, th_trans=curr_t, use_registed_beta=True)
        curr_r = unit_quaternion_to_matrix(mano_axisang2quat(curr_coeff[:,:3]))
        curr_theta = curr_coeff[:,3:]
        return torch.tensor(final_kp), curr_theta, curr_r.squeeze(0), curr_t