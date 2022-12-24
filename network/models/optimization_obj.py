import numpy as np
import torch
from pose_utils.rotations import unit_quaternion_to_matrix
from DeepSDF import Decoder
import os
import cv2
import open3d as o3d
import torch.nn as nn
import sys 
from os.path import join as pjoin 
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..', '..', '..', 'Curriculum-DeepSDF'))
import deep_sdf

if os.path.isfile('/data1/h2o_data/HO3D/CatPose2InsPose.npy'):
    change = np.load('/data1/h2o_data/HO3D/CatPose2InsPose.npy', allow_pickle=True).item()
else:
    change = np.load('/data/h2o_data/HO3D/CatPose2InsPose.npy', allow_pickle=True).item()

class soft_L1(nn.Module):
    def __init__(self):
        super(soft_L1, self).__init__()

    def forward(self, input, target, eps=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret

def CatCS2InsCS(x, normalization_params, instance):
    R = change[instance]['rotation']
    T = change[instance]['translation']
    if isinstance(x, np.ndarray):
        result = (x + normalization_params['offset'])*normalization_params['scale']
        result = np.matmul(result, R.transpose(-1,-2))+T
    elif  isinstance(x, torch.Tensor):
        result = (x + torch.FloatTensor(normalization_params['offset']).to(x.device))*torch.FloatTensor(normalization_params['scale']).to(x.device)
        result = torch.matmul(result, torch.FloatTensor(R).to(x.device).transpose(-1,-2))+torch.FloatTensor(T).to(x.device)
    return result

def InsCS2CatCS(x, normalization_params, instance):
    R = change[instance]['rotation']
    T = change[instance]['translation']
    if isinstance(x, np.ndarray):
        result = np.matmul(x-T, R)
        result = result / normalization_params['scale'] - normalization_params['offset']
    elif  isinstance(x, torch.Tensor):
        result = torch.matmul(x-torch.FloatTensor(T).to(x.device), torch.FloatTensor(R).to(x.device))
        result = result / torch.FloatTensor(normalization_params['scale']).to(x.device) - torch.FloatTensor(normalization_params['offset']).to(x.device)
    return result

def get_RT(instance):
    if instance not in change.keys():
        R = np.eye(3)
        T = np.zeros(3)
    else:
        R = change[instance]['rotation']
        T = change[instance]['translation']
    return R, T

def world2point2D(xyz, fx, fy, cx, cy):
    '''
        xyz: B, N, 3
        point_2D B, N, 2
    '''
    B, N, _ = xyz.shape
    x = (xyz[..., 0] / xyz[..., 2] * fx + cx).reshape(B, N, 1)
    y = (xyz[..., 1] / xyz[..., 2] * fy + cy).reshape(B, N, 1)
    point_2D = torch.cat([y, x], dim=-1).float()
    return point_2D # B, N, 2


class gf_optimize_obj():
    def __init__(self, cfg):

        # important parameters
        self.particle_size = 2048 
        self.iteration = 10 
        self.scaling_coefficient1 = 0.02
        self.scaling_coefficient2 = 2
        self.volume_size = 201  
        self.voxel_scale = 0.002
        self.beta = 0.9 # don't important

        self.update_shape_flag = cfg['network']['updateshape']
        self.device = cfg['device']
        latent_size = 256
        self.SDFDecoder = Decoder(latent_size, **cfg['network']["NetworkSpecs"])
        self.SDFDecoder = torch.nn.DataParallel(self.SDFDecoder)

        # pre-define sdf volume
        self.volume_ind = torch.arange(self.volume_size**3)[:,None].repeat(1,3)
        self.volume_ind[:, 2] = self.volume_ind[:, 2] % self.volume_size
        self.volume_ind[:, 1] = self.volume_ind[:, 1] // self.volume_size % self.volume_size
        self.volume_ind[:, 0] = self.volume_ind[:, 0] // self.volume_size // self.volume_size
        self.volume_ind = (self.volume_ind - self.volume_size//2) * self.voxel_scale
        self.volume_ind = self.volume_ind.to(self.device)
        self.sdf_weight = cfg['optimization']['sdf_weight']
        
        # pre-sample particles
        mean = np.zeros(6)
        cov = np.eye(6) 
        self.pre_sampled_particle = np.random.multivariate_normal(mean, cov, self.particle_size)
        self.pre_sampled_particle[0,:] = 0
        self.pre_sampled_particle = torch.FloatTensor(self.pre_sampled_particle).to(self.device)

        self.dataset_name = cfg['data_cfg']['dataset_name']
        

    def load_obj(self, obj_info, instance, init_pose, init_pc):
        latent_code_pth, normalization_param, saved_model_pth,_,_ = obj_info 
        saved_model_state = torch.load(saved_model_pth)

        self.SDFDecoder.load_state_dict(saved_model_state["model_state_dict"])
        SDFDecoder = self.SDFDecoder.module.to(self.device)

        print('load sdf code from %s' % latent_code_pth)
        self.latent_code = torch.load(latent_code_pth)[0][0].to(self.device) # 1, 1, L
        

        if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB': # bottle, can, box, bowl
            self.ins_volume_ind = CatCS2InsCS(self.volume_ind, normalization_param, instance)
        elif self.dataset_name == 'newShapeNet':
            self.ins_volume_ind = (self.volume_ind + torch.FloatTensor(normalization_param['offset']).to(self.device))*torch.FloatTensor(normalization_param['scale']).to(self.device)
        else:
            raise NotImplementedError
        voxelsdf_pth = latent_code_pth.replace('Codes', 'voxelsdf').replace('.pth', '.npy')
        # if os.path.isfile(voxelsdf_pth):
        #     print('load from ',voxelsdf_pth)
        #     sdfdata = np.load(voxelsdf_pth, allow_pickle=True)
        #     # if sdfdata['size'] != self.volume_size or sdfdata['scale'] != self.voxel_scale:
        #         # print('wrong!')
        #         # exit(1)
        #     self.sdf_volume = torch.FloatTensor(sdfdata).to(self.device)
        # else:
        if True: 
            with torch.no_grad():
                piece = 10
                all_length = self.ins_volume_ind.shape[0]
                length = all_length // piece + 1
                self.sdf_volume = torch.zeros((all_length, 1), dtype=torch.float16).cuda()
                for i in range(piece):
                    latent_inputs = self.latent_code.expand(min(all_length, (i+1)*length)-i*length, -1)
                    inputs = torch.cat([latent_inputs, self.ins_volume_ind[i*length:min(all_length, (i+1)*length)]], 1)
                    self.sdf_volume[i*length:min(all_length, (i+1)*length)] = SDFDecoder(inputs)
                self.sdf_volume = self.sdf_volume.reshape(self.volume_size,self.volume_size,self.volume_size) / normalization_param['scale'][0]       #[V^3, 1]
            os.makedirs(os.path.dirname(voxelsdf_pth), exist_ok=True)
            # np.save(voxelsdf_pth[:-4], self.sdf_volume.cpu().numpy())
            # np.save(voxelsdf_pth[:-4], {'sdf':self.sdf_volume.cpu().numpy(),'size':self.volume_size, 'scale': self.voxel_scale})
            # print('save to ', voxelsdf_pth)
        # inputs = torch.cat([latent_inputs, self.ins_volume_ind], 1)
        # with torch.no_grad():
        #     self.sdf_volume = SDFDecoder(inputs).reshape(self.volume_size,self.volume_size,self.volume_size) / normalization_param['scale'][0]       #[V^3, 1]
        self.latent_code_pth = latent_code_pth
        os.system(f"cp {self.latent_code_pth} {self.latent_code_pth.replace('.pth', '_update.pth')}")
        self.saved_model_pth = saved_model_pth
        self.obj_merged_pc = torch.matmul(init_pc.float() - init_pose['translation'].squeeze(-1), init_pose['rotation'].squeeze(1)).cuda()
        self.obj_merged_pc = CatCS2InsCS(self.obj_merged_pc, normalization_param, instance)
        camera = torch.matmul(torch.zeros((1,1,3)) - init_pose['translation'].squeeze(-1), init_pose['rotation'].squeeze(1)).cuda()
        camera = CatCS2InsCS(camera, normalization_param, instance)
        self.obj_merged_normal = self.estimate_normal(self.obj_merged_pc, camera)
        self.obj_merge_num = 1
        self.normalization_param = normalization_param
        self.instance = instance
        return 
    
    def load_obj_oracle(self, instance):
        import trimesh
        gt_obj_mesh = trimesh.load(f'/data/h2o_data/HO3D/models/{instance}/textured_simple.obj')
        gt_obj_faces = torch.LongTensor(gt_obj_mesh.faces).reshape(-1, 3).cuda()
        gt_obj_verts = torch.FloatTensor(gt_obj_mesh.vertices).reshape(1, -1, 3).cuda()

        import kaolin 
        from kaolin.ops.mesh import index_vertices_by_faces
        face_vertices = index_vertices_by_faces(gt_obj_verts, gt_obj_faces)
        dis,_,_ = kaolin.metrics.trianglemesh.point_to_mesh_distance(self.volume_ind.unsqueeze(0), face_vertices)
        sign = kaolin.ops.mesh.check_sign(gt_obj_verts, gt_obj_faces, self.volume_ind.unsqueeze(0), hash_resolution=4096)
        sdf = torch.sqrt(dis) * (-2*sign+1)
        np.save( f'{instance}_oracle', sdf.cpu().numpy())
        
        sdf = torch.clamp(sdf, -0.1, 0.1)
        self.sdf_volume  = sdf.reshape(self.volume_size,self.volume_size,self.volume_size)
        print('finish loading')
        return 

    def get_silhouette_loss(self, obj):
        pred_2D = world2point2D(obj, self.proj['fx'][0], self.proj['fy'][0], self.proj['cx'][0], self.proj['cy'][0])   #[B, N, 2]
        index1 = torch.clamp(pred_2D[...,0].long(), 0, self.h-1)
        index2 = torch.clamp(pred_2D[...,1].long(), 0, self.w-1)
        silhouette_loss = self.gt_mask[index1, index2]
        silhouette_loss = silhouette_loss.sum(dim=-1) / pred_2D.shape[1]
        return silhouette_loss

    def load_gt_mask(self, category, file_name):
        if self.dataset_name == 'HO3D':
            silhouette_pth = '/data/h2o_data/HO3D/pred_mask/ensemble/%s/%s.png' % (file_name.split('/')[0], file_name.split('/')[1])
            mask = (cv2.imread(silhouette_pth) / 70)[:240] # 0:obj,1:hand,2back
            mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)[:, :, 0]
            self.gt_mask = (mask == 2)
            self.gt_mask = torch.tensor(self.gt_mask).to(self.device)
        elif self.dataset_name == 'DexYCB':
            silhouette_pth = pjoin('/data/h2o_data/DexYCB/tarfolder/%s/%s/%s/labels_%s.npz' % (file_name.split('+')[0],file_name.split('+')[1],file_name.split('+')[2],file_name.split('+')[3]))
            color_pth = silhouette_pth.replace('labels', 'color').replace('npz', 'jpg')
            rgbimg = cv2.imread(color_pth)
            maskimg = np.load(silhouette_pth)['seg']
            self.gt_mask = maskimg==0
            self.rgbimg = rgbimg * self.gt_mask[:,:,None]
            self.gt_mask = torch.tensor(self.gt_mask).to(self.device)
        elif self.dataset_name == 'new_shapenet':
            silhouette_pth = '/data/h2o_data/new_sim_dataset/render/img/%s/seq/%s/mask.png' % (category, file_name)
            maskimg = cv2.imread(silhouette_pth)
            self.gt_mask = maskimg.sum(axis=-1) == 0
            self.gt_mask = torch.tensor(self.gt_mask).to(self.device)
        else:
            raise NotImplementedError
        return 

    def Distance(self,V):
        with torch.no_grad():
            bboxMin = - 0.2
            bboxRes = self.volume_size
            stride =  self.voxel_scale
            distance = self.sdf_volume.reshape(-1)

            x = (V[:,0] - bboxMin) / stride
            y = (V[:,1] - bboxMin) / stride
            z = (V[:,2] - bboxMin) / stride
            x = torch.clamp(x, 0, bboxRes-1)
            y = torch.clamp(y, 0, bboxRes-1)
            z = torch.clamp(z, 0, bboxRes-1)


            xIdx = x.data.long()
            yIdx = y.data.long()
            zIdx = z.data.long()

            x.data -= xIdx
            y.data -= yIdx
            z.data -= zIdx
            i000 = (xIdx * bboxRes + yIdx) * bboxRes + zIdx
            i001 = i000 + 1
            i010 = i000 + bboxRes
            i011 = i001 + bboxRes
            i100 = i000 + bboxRes * bboxRes
            i101 = i001 + bboxRes * bboxRes
            i110 = i010 + bboxRes * bboxRes
            i111 = i011 + bboxRes * bboxRes
            clamp_value = len(distance)
            i000 = torch.clamp(i000, 0, clamp_value - 1).long()
            i001 = torch.clamp(i001, 0, clamp_value - 1).long()
            i010 = torch.clamp(i010, 0, clamp_value - 1).long()
            i011 = torch.clamp(i011, 0, clamp_value - 1).long()
            i100 = torch.clamp(i100, 0, clamp_value - 1).long()
            i101 = torch.clamp(i101, 0, clamp_value - 1).long()
            i110 = torch.clamp(i110, 0, clamp_value - 1).long()
            i111 = torch.clamp(i111, 0, clamp_value - 1).long()
            dis = ((distance[i000] * (1 - z) + distance[i001] * z) * (1 - y)\
                + (distance[i010] * (1 - z) + distance[i011] * z) * y) * (1 - x)\
                + ((distance[i100] * (1 - z) + distance[i101] * z) * (1 - y)\
                + (distance[i110] * (1 - z) + distance[i111] * z) * y) * x
            dis = torch.clamp(dis,-0.05,0.05) 
        return dis

    def evaluate(self, pcld, r, t):   #B=particle_size N=point_size
        # chamfer loss
        _, N, _ = pcld.shape
        pcld_flat = torch.matmul((pcld - t.transpose(-1,-2)), r).reshape(-1, 3)  
        queried_sdf = self.Distance(pcld_flat).reshape(-1, N)
        sdf_energy = torch.mean(queried_sdf.abs(), dim=-1)
        energy = sdf_energy * self.sdf_weight
        return energy, sdf_energy

    def update_seach_size(self, tsdf, mean_transform):
        s = mean_transform.abs() + 1e-3
        search_size = tsdf * self.scaling_coefficient2 * s / s.norm() + 1e-3   
        return search_size
    
    def optimize(self, pcld, init_obj_pose, category, file_name, projection):
        rotation, translation = init_obj_pose['rotation'].float(), init_obj_pose['translation'].float()
        self.proj = projection
        self.w = projection['w'][0]
        self.h = projection['h'][0]
        if self.silhouette:
            self.load_gt_mask(category, file_name)
        search_size = self.scaling_coefficient1
        prev_search_size = search_size
        count = 0
        prev_success_flag = True
        pcld = torch.FloatTensor(pcld.float()).to(self.device)

        while (True):
            if count == self.iteration:
                break 

            # get delta pose from pre-sampled particles
            sample_part = self.pre_sampled_particle*search_size
            sample_qw = torch.sqrt(1-sample_part[:,0]**2-sample_part[:,1]**2-sample_part[:,2]**2).unsqueeze(1)
            sample = torch.cat([sample_qw, sample_part],dim=1)
            sample_r = unit_quaternion_to_matrix(sample[:, :4])
            new_r =  torch.matmul(rotation, sample_r) 
            new_t = translation + sample[:, 4:, None]

            # evaluate each particle
            energy, sdf_energy = self.evaluate(pcld, new_r, new_t)    #[B]

            # filter good particles
            origin_energy = energy[0]
            better_mask = energy < origin_energy  #[B]
            weight = (origin_energy - energy) * better_mask      #[B]
            weight_sum = weight.sum() + 1e-5
            if torch.any(better_mask):
                mean_sdf = (sdf_energy * weight).sum() / weight_sum
                success_flag = True  
            else:
                mean_sdf = sdf_energy[0]  
                success_flag = False 

            #update R, T
            if success_flag:
                mean_transform = (sample * weight.unsqueeze(1)).sum(dim=0, keepdim=True) / weight_sum    #[1, 7]
                mean_transform[:, :4] /= (mean_transform[:,:4].norm() + 1e-5)
                rotation = torch.matmul(rotation, unit_quaternion_to_matrix(mean_transform[:, :4])) 
                translation = translation + mean_transform[:, 4:, None]
            else:
                mean_transform = torch.zeros((1,7), device=self.device)
            
            # update search size
            search_size = self.update_seach_size(mean_sdf, mean_transform[:,1:])
            if prev_success_flag and success_flag:
                search_size = self.beta * search_size + (1-self.beta)*prev_search_size
                prev_search_size = search_size
            elif success_flag:
                prev_search_size = search_size
            prev_success_flag = success_flag

            count += 1
        ret_dict = {
            'rotation': rotation,
            'translation': translation.reshape(1, 3, 1),
        }
        
        final_pcld = torch.matmul((pcld - ret_dict['translation'].transpose(-1,-2)), ret_dict['rotation'])

        if self.update_shape_flag:
            pcld_flat = final_pcld.reshape(-1,3)
            N = pcld_flat.shape[0]
            queried_sdf = self.Distance(pcld_flat)
            good_mask = torch.where(queried_sdf.abs()<0.02)  
            final_pcld = final_pcld[:,good_mask[0]] 
            # good_mask = queried_sdf.abs()<0.02   
            # final_pcld = final_pcld * good_mask[...,None]
            final_pcld = CatCS2InsCS(final_pcld, self.normalization_param, self.instance)
            camera_center = torch.matmul(torch.zeros((1,1,3)).cuda() - ret_dict['translation'].transpose(-1,-2), ret_dict['rotation'])
            camera_center = CatCS2InsCS(camera_center, self.normalization_param, self.instance)

            self.obj_merge_num += 1
            choose_num = self.obj_merged_pc.shape[1] // self.obj_merge_num
            new_pc_ind = np.random.permutation(final_pcld.shape[1])[:choose_num]
            old_pc_ind = np.random.permutation(self.obj_merged_pc.shape[1])[:self.obj_merged_pc.shape[1]-choose_num]

            final_pcld = final_pcld[:, new_pc_ind]
            self.obj_merged_pc = torch.cat([self.obj_merged_pc[:,old_pc_ind], final_pcld], dim=1)
            new_normal = self.estimate_normal(final_pcld, camera_center)
            self.obj_merged_normal = np.concatenate([self.obj_merged_normal[old_pc_ind], new_normal], axis=0)

            # update shape
            if self.obj_merge_num % 10 == 0:
                self.update_shape()
        return ret_dict

    def estimate_normal(self, pc, camera):
        pc = pc[0].cpu().numpy()
        camera = camera[0].cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        normals = (2 * (normals * (camera - pc) > 0) - 1) * normals
        return normals 

    def update_shape(self, num_iterations=100, clamp_dist=0.2, lr=1e-3, l2reg=True):
        latent_init = torch.tensor(self.latent_code, device="cuda")
        latent_init.requires_grad = True
        decoderm = self.SDFDecoder.module.cuda()
        # decoderm.train()
        # optimizer = torch.optim.Adam([{'params':latent_init}, {'params':decoderm.parameters(), 'lr':1e-5}], lr=lr)
        decoderm.eval()
        optimizer = torch.optim.Adam([{'params':latent_init}], lr=lr)
        loss_l1 = soft_L1()

        obj_pc = self.obj_merged_pc[0].cpu().numpy()
        normals = self.obj_merged_normal
        zero_sdf = torch.zeros((obj_pc.shape[0], 1)).cuda()
       
        for e in range(num_iterations):
            if e >= num_iterations // 2:
                optimizer.param_groups[0]['lr'] = lr / 2
            miu_pos = np.random.rand(obj_pc.shape[0], 1)*0.1 
            miu_neg = np.random.rand(obj_pc.shape[0], 1)*0.05 
            outside = obj_pc + normals * miu_pos
            inside = obj_pc - normals * miu_neg
            sdf_gt = torch.cat([zero_sdf+torch.FloatTensor(miu_pos).cuda(), zero_sdf, zero_sdf-torch.FloatTensor(miu_neg).cuda()], dim=0)
            xyz = np.concatenate([outside, obj_pc, inside], axis=0)
            xyz = torch.FloatTensor(xyz).cuda()
            sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)

            optimizer.zero_grad()
            latent_inputs = latent_init.expand(xyz.shape[0], -1)
            inputs = torch.cat([latent_inputs, xyz], 1).cuda()
            pred_sdf = decoderm(inputs)

            pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
            loss = loss_l1(pred_sdf, sdf_gt)
            loss *= (1+ 0.5 * torch.sign(sdf_gt) * torch.sign(sdf_gt-pred_sdf))
            loss = torch.mean(loss)
            if l2reg:
                loss += 1e-4 * torch.mean(latent_init.pow(2))
            loss.backward()
            optimizer.step()
        self.latent_code = latent_init.detach()
        torch.save(latent_init.unsqueeze(0), self.latent_code_pth.replace('.pth', '_update.pth'))

        with torch.no_grad():
            latent_inputs = self.latent_code.expand(self.ins_volume_ind.shape[0], -1)
            inputs = torch.cat([latent_inputs, self.ins_volume_ind], 1)
            # piece = 1
            # length = inputs.shape[0] // piece + 1
            # self.sdf_volume = torch.zeros((inputs.shape[0], 1), dtype=torch.float16).cuda()
            # for i in range(piece):
            #     self.sdf_volume[i*length:min(inputs.shape[0], (i+1)*length)] = decoderm(inputs[i*length:min(inputs.shape[0], (i+1)*length)])
            # self.sdf_volume = self.sdf_volume.reshape(self.volume_size,self.volume_size,self.volume_size) / self.normalization_param['scale'][0]       #[V^3, 1]
            self.sdf_volume = decoderm(inputs).reshape(self.volume_size,self.volume_size,self.volume_size) / self.normalization_param['scale'][0]       #[V^3, 1]
        return

    def sdf2mesh(self, mesh_filename):
        with torch.no_grad():
            decoderm = self.SDFDecoder.module.cuda()
            deep_sdf.mesh.create_mesh(
                decoderm, self.latent_code, mesh_filename.replace('.ply', ''), N=128, max_batch=int(2 ** 18)
            )
        return 

