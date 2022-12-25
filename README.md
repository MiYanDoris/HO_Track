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
  conda env create -n hotrack python=3.7
  conda activate hotrack
  ```

+ Install pytorch and other dependencies.

  ```bash
 pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
 
  ```

+ Compile the CUDA code for PointNet++ backbone.

  ```bash
  cd network/models/pointnet_lib
  python setup.py install
  ```

+ Download MANO pickle data-structures and save it to ```third_party/mano/models``` following [Manopth](https://github.com/hassony2/manopth).


## Dataset

+ Download SimGrasp dataset in (TODO).

+ Download HO3D dataset (version 3) in [here](https://cloud.tugraz.at/index.php/s/z8SCsWCYM3YcQWX?). 

+ Download DexYCB dataset. 


## Running
+ Somethings you should notice
  
  + All configs are in `configs/all_config`. You need to change root_list and mano_path_lst in `configs/config.py`. 
  
+ To train HandTrackNet. The test results reported during training is based on single-frame instead of tracking a sequence.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/train.py --config handtracknet_train_SimGrasp.yml
  ```

+ To track hand in a sqeuence.
  ```bash
    CUDA_VISIBLE_DEVICES=0 python network/test.py --config handtracknet_test_SimGrasp.yml --num_worker 0
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



