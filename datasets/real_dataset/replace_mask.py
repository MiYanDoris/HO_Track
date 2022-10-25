import os


root_dir = '/mnt/data/hewang/h2o_data/C5'
obj_lst = os.listdir(root_dir)
obj_lst.sort()
for obj_id in obj_lst:
    obj_dir = os.path.join(root_dir, obj_id)
    scene_lst = os.listdir(obj_dir)
    scene_lst.sort()
    for scene in scene_lst:
        scene_dir = os.path.join(obj_dir, scene)
        layout_lst = os.listdir(scene_dir)
        layout_lst.sort()
        for layout in layout_lst:
            layout_dir = os.path.join(scene_dir, layout)
            task_lst = os.listdir(layout_dir)
            task_lst.sort()
            for task in task_lst:
                task_dir = os.path.join(layout_dir, task)
                for i in range(30):
                    correct_mask_pth = os.path.join(task_dir, '2Dseg/mask', '%05d.png' % (i * 10))
                    target_mask_pth = correct_mask_pth.replace('/C5', '/HOI4D/C5')
                    os.system('cp %s %s' % (correct_mask_pth, target_mask_pth))