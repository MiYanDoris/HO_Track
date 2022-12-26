# Tracking and Reconstructing Hand Object Interactions from Point Cloud Sequences in the Wild

## Introduction

This is the PyTorch implementation of [our paper](https://arxiv.org/abs/2209.12009). This repository is still under construction.


## Installation

+ Our code has been tested with
  + Ubuntu 20.04
  + CUDA 11.7
  + Python 3.8
  + PyTorch 1.9.1 (NOTE: If PyTorch version>1.10, there are bugs when compiling CUDA code in ```pointnet_lib```)

+ We recommend using [Anaconda](https://www.anaconda.com/) to create an environment, by running the following:

  ```bash
  conda env create -n hotrack python=3.8
  conda activate hotrack
  ```

+ Install pytorch and other dependencies.

  ```bash
  pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -r requirements.txt
  ```

+ Compile the CUDA code for PointNet++ backbone.

  ```bash
  cd network/models/pointnet_lib
  python setup.py install
  ```



## Dataset

+ Download MANO pickle data-structures and save it to ```third_party/mano/models``` following [Manopth](https://github.com/hassony2/manopth#download-mano-pickle-data-structures). You also need to install Manopth if you want to play with DexYCB dataset.

+ Download SimGrasp dataset and our pre-computed SDF models in (TODO). We use [Curriculum-DeepSDF](https://github.com/haidongz-usc/Curriculum-DeepSDF) to obtain DeepSDF models taking the observed point clouds at frame 0 as input.

+ Download HO3D dataset (version 3) from [their official website](https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX?). Unzip 

+ Download DexYCB dataset from [their official website](https://dex-ycb.github.io/).

### Dataset Folder Structure

<details>
<summary><b>See here for the organization of dataset folders.</b> </summary> 
<p>

 ```bash
    data
    ├── SimGrasp
    │   ├── img # raw RGB and depth, which is not necessary for training and testing
    │   ├── objs
    │   ├── mask # Only used in the hand optimization stage in the full pipeline
    │   ├── preproc
    │   ├── splits
    │   └── SDF
    ├── YCB
    │   ├── CatPose2InsPose.npy 
    │   ├── models # Download from the DexYCB dataset
    │   └── SDF
    ├── HO3D
    │   ├── calibration 
    │   ├── train # Contain both HO3D_v3.zip and HO3D_v3_segmentations_rendered.zip
    │   ├── splits
    │   └── SDF
    ├── DexYCB 
    │   ├── 20200709-subject-01
    │   ├── ...
    │   ├── 20201022-subject-10
    │   ├── calibration
    │   ├── splits
    │   └── SDF
    └── exps				
    ```
</p>
</details>


## Running
+ Somethings you should notice
  
  + All configs are in `configs/all_config`. You need to change root_list and mano_path_lst in `configs/config.py`. 
  
+ To train HandTrackNet. The test results reported during training is based on single-frame instead of tracking a sequence.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/train.py --config handtracknet_train_SimGrasp.yml
  ```

+ To track hand using HandTrackNet in a sqeuence.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/test.py --config handtracknet_test_SimGrasp.yml --num_worker 0
  ```

+ Our full pipeline.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/test.py --config objopt_test_HO3D.yml --num_worker 0 --save # 1. track object and save results
    CUDA_VISIBLE_DEVICES=0 python network/test.py --config handopt_test_HO3D.yml --num_worker 0 # 2. track hand using saved object pose
  ```

<!-- 
## SimGrasp generation
+ Follow instructions to install [the melodic version of ROS](http://wiki.ros.org/melodic/Installation) + [graspit interface](https://github.com/graspit-simulator/graspit_interface) + [mano_grasp](https://github.com/lwohlhart/mano_grasp)
  + **Note** that **graspit** only support **melodic** version of ROS on ubuntu **18.04**

+ NOCS dataset: [Download from the original NOCS dataset](https://github.com/hughw19/NOCS_CVPR2019#datasets): 

  ```
  python cp_nocs_objs.py -c bottle      #first copy the obj files for prepare_objects. It will also resize the objects to be like in real.
  cd mano_grasp/mano_grasp
  python prepare_objects.py --models_folder /home/hewang/Desktop/data/jiayi/h2o_data/objs/bottle --file_out NOCS_bottle.txt  --scales 1000
  roslaunch graspit_interface graspit_interface.launch   #in another cmd
  python generate_grasps.py --models_file NOCS_bottle.txt --path_out /home/hewang/Desktop/data/jiayi/h2o_data/grasps/bottle -n 30 -g 10
  
  cd preproc_grasps_data
  python remove_duplicate_grasp.py -c bottle
  python save_hand_mesh -c bottle
  python my_render.py -c bottle
  ```
   -->



