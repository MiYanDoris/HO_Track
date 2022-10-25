from http.client import CONTINUE
import numpy as np
import tqdm
import os

root_dir = '/mnt/data/hewang/h2o_data/HOI4D'
task_list = []
txt_pth = '/mnt/data/hewang/h2o_data/HOI4D/C5/bottle_refined_pose.txt'
with open(txt_pth, "r", errors='replace') as fp:
    lines = fp.readlines()
    for i in lines:
        task_list.append(i.strip())

for task in tqdm.tqdm(task_list):
    for i in range(300):
        if i % 30 != 0:
            CONTINUE
        npy_pth = os.path.join(root_dir, task, 'preprocess/%03d.npy' % i)
        pcld_pth = os.path.join(root_dir, 'hand_pcld', task, '%03d.npy' % i)
        dir_name = os.path.dirname(pcld_pth)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        dict = np.load(npy_pth, allow_pickle=True).item()
        pcld = dict['hand_pcld']
        if len(pcld) > 512:
            sample_idx = np.arange(len(pcld))
            np.random.shuffle(sample_idx)
            sample_idx = sample_idx[:512]
            pcld = pcld[sample_idx]
        np.save(pcld_pth, pcld)