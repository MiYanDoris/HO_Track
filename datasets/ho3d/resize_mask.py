import cv2
import os

seq_lst = os.listdir('/mnt/data/hewang/h2o_data/HO3D/mask/')
for seq in seq_lst:
    if seq == '.DS_Store':
        continue
    frame_lst = os.listdir('/mnt/data/hewang/h2o_data/HO3D/mask/' + seq + '/seg/')
    mask_aligned_dir = '/mnt/data/hewang/h2o_data/HO3D/mask/' + seq + '/aligned_mask/'
    for frame in frame_lst:
        if os.path.isfile(mask_aligned_dir + frame):
            continue
        if frame == '.DS_Store':
            continue
        mask_pth = '/mnt/data/hewang/h2o_data/HO3D/mask/' + seq + '/seg/' + frame
        mask = cv2.imread(mask_pth)
        mask_large = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        if not os.path.exists(mask_aligned_dir):
            os.mkdir(mask_aligned_dir)
        cv2.imwrite(mask_aligned_dir + frame, mask_large)
        print('save to ', mask_aligned_dir + frame)