import os
import numpy as np
import cv2
import argparse
import os
import os.path
import sys
import cv2
import numpy as np
from PIL import Image
from numpy.core.fromnumeric import choose
import scipy.io as scio
import argparse
import matplotlib.pyplot as plt
import tqdm
import collections
import yaml
import open3d as o3d
import torch

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)
# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map

def fill_in_multiscale(depth_map, max_depth=8.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 1.0)
    valid_pixels_med = (depths_in > 1.0) & (depths_in <= 2.0)
    valid_pixels_far = (depths_in > 2.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict

def fill_missing(
        dpt, cam_scale, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt * cam_scale
    return dpt

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1].type(torch.long)
    return centroids

def dpt_2_cld(dpt, cam_scale, K):
    H, W = dpt.shape

    xmap = np.array([[j for i in range(W)] for j in range(H)])
    ymap = np.array([[i for i in range(W)] for j in range(H)])
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    
    if len(choose) < 1:
        return None, None
    
    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32) 
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)
    
    pt2 = dpt_mskd / cam_scale
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)  # position in camera reference
    
    return cld, choose

def get_pc_img(dpt, img, K, mask, cam_scale=1000):
    invalid = (mask == 0)
    dpt[invalid] = 0

    cld, choose = dpt_2_cld(dpt, cam_scale, K)
    img = img.reshape(-1, 3)[choose]

    o3dpc = o3d.geometry.PointCloud() 
    o3dpc.points = o3d.utility.Vector3dVector(cld) 
    _, ind = o3dpc.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
    img = img[ind]
    cld = cld[ind]

    fps_idx = farthest_point_sample(torch.from_numpy(np.expand_dims(cld, axis=0)), 4096)
    fps_cld = cld[fps_idx[0]]
    fps_img = img[fps_idx[0]]
    color_cld = np.concatenate((fps_cld, fps_img), axis=1)
    return color_cld

def get_pc(dpt, K, id, dir_pth, cam_scale=1000):
    cld, choose = dpt_2_cld(dpt, cam_scale, K)
    if not os.path.exists(dir_pth):
        os.system('mkdir -p {}'.format(dir_pth))
    
    np.savetxt(os.path.join(dir_pth, id + '.txt'), cld)

def get_pc_from_o3d(dpt, color_intrin):
    height, width = dpt.shape
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
            dpt, o3d.camera.PinholeCameraIntrinsic(width=width,
                                                                 height=height,
                                                                 fx=color_intrin[0][0],
                                                                 fy=color_intrin[1][1],
                                                                 cx=color_intrin[0][2],
                                                                 cy=color_intrin[1][2]))
    pcld = np.asarray(pcd.points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Recorder. Please select one of the optional arguments")
    parser.add_argument(
        "--dir_path",
        default='/mnt/data/hewang/h2o_data/real_dataset_1/C5/N06/S122/s3/T1/',
        help="Recording save color and depth images into realsense folder")
    args = parser.parse_args()

    # load camera parameters
    color_intrin = np.load('/mnt/data/hewang/h2o_data/real_dataset_1/intrinsics/color_intrin.npy')
    depth_intrin = np.load('/mnt/data/hewang/h2o_data/real_dataset_1/intrinsics/depth_intrin.npy')
    if not os.path.exists(os.path.join(args.dir_path, 'pcld')):
        os.mkdir(os.path.join(args.dir_path, 'pcld'))
    for i in range(50):
        # number = 6 * i
        number = 20
        depth_file_name = os.path.join(args.dir_path, 'align_depth', '%d.png' % number)
        depth_map = cv2.imread(depth_file_name, cv2.IMREAD_UNCHANGED)
        print(depth_map[500:600:, 900:1000])
        print(np.max(depth_map))
        exit(0)

        mask = cv2.imread(os.path.join(args.dir_path, '2Dseg/mask', '%05d.png' % number))
        mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        # mask = mask[:, :, 2]
        mask = np.sum(mask, axis=-1)

        color_filename = os.path.join(args.dir_path, 'align_image', '%d.jpg' % number)
        color_map = cv2.imread(color_filename)
        color_pcld = get_pc_img(depth_map, color_map, color_intrin, mask)
        np.savetxt('color_pcld.txt', color_pcld)
        exit(0)
        # np.save(os.path.join(args.dir_path, 'pcld', '%05d.npy' % number), color_pcld)
