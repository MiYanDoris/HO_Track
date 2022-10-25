import os
import tqdm

root_dir = '/mnt/data/hewang/h2o_data/HOI4D'
task_list = []
txt_pth = '/mnt/data/hewang/h2o_data/HOI4D/C5/bottle_refined_pose.txt'
with open(txt_pth, "r", errors='replace') as fp:
    lines = fp.readlines()
    for i in lines:
        task_list.append(i.strip())

for task in tqdm.tqdm(task_list):
    source_name = os.path.join('/mnt/data/hewang/h2o_data', task)
    target_name = os.path.join(root_dir, task, 'hand_pose_refined')
    file_lst = os.listdir(source_name)
    for file in file_lst:
        file_pth = os.path.join(source_name, file)
        target_file_pth = os.path.join(target_name, file)
        if not os.path.exists(os.path.dirname(target_file_pth)):
            os.mkdir(os.path.dirname(target_file_pth))
        os.system('cp %s %s' % (file_pth, target_file_pth))