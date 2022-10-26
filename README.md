# Tracking and Reconstructing Hand Object Interactions from Point Cloud Sequences in the Wild

## Introduction

This is the PyTorch implementation of [our paper](https://arxiv.org/abs/2209.12009). This repository is still under construction.


## Installation

+ Our code has been tested with
  + Ubuntu 18.04
  + CUDA 11.0
  + Python 3.7.7
  + PyTorch 1.6.0

+ We recommend using [Anaconda](https://www.anaconda.com/) to create an environment, by running the following:

  ```bash
  conda env create -n hotrack python=3.7
  conda activate hotrack
  ```

+ Install dependencies.

  ```bash
  pip install -r requirements.txt
  ```

+ Compile the CUDA code for PointNet++ backbone.

  ```bash
  cd network/models/pointnet_lib
  python setup.py install
  ```


## Running
+ Somethings you should notice
  + We leave some code about HOI4D for your convenience but we don't use them for a long time and there must be some bugs. You'd better check carefully.
  + We reimplement ManoLayer in `network/models/our_mano.py` since original version has a small problem that the hand root is not in origin even if translation=0. This is very important and please be careful about ManoLayer. If you want to use the original version, just set original_version=True when forward.
  + All configs are in `configs/all_config`. You need to change root_list and mano_path_lst in `configs/config.py`. Download Mano models following [this](https://github.com/hassony2/manopth).
  
+ To train HandTrackNet. The test results reported during training is based on single-frame instead of tracking a sequence.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/train.py --config handtracknet_train_SimGrasp.yml
  ```

+ To track hand in a sqeuence. There is some problem if num_worker!=0.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/test.py --config handtracknet_test_HO3D.yml --num_worker 0
  ```

+ useful parse_args (there also may be some bugs): 
  + --debug: use model.visualize() and show plt figures
  + --debug_save: use model.visualize() and save plt figures to network/debug/..
  + --save:  save tracking results for generating .gif file



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
  
+ Sapien dataset: [Download from object URDF models](http://download.cs.stanford.edu/orion/captra/sapien_urdf.tar)

  ```
  cd partnet_articulated_obj
  python urdf2obj.py -c laptop
  cd mano_grasp/mano_grasp
  python prepare_objects.py --models_folder /home/hewang/Desktop/data/jiayi/h2o_data/objs/laptop --file_out sapien_laptop.txt  --scales 1000
  roslaunch graspit_interface graspit_interface.launch    #in another cmd
  python generate_grasps.py --models_file sapien_laptop.txt --path_out /home/hewang/Desktop/data/jiayi/h2o_data/grasps/laptop -n 30 -g 20 --valid
  
  cd preproc_grasps_data
  python remove_duplicate_grasp.py -c laptop
  python save_hand_mesh -c laptop
  cd partnet_articulated_obj
  python sapien_render.py -c laptop
  ```



