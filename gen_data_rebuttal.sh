#!/bin/bash

# setup env variables
export POINTNAV_VO_ROOT=$PWD && \
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \

# # apply habitat sim fix
# cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
# cd /home/memmel/habitat-sim && \
# pip install -r requirements.txt && \
# python setup.py install --headless

# go to root
cd ${POINTNAV_VO_ROOT} && \

# login to git
sh login_git.sh

pip install einops

# exec script
python ${POINTNAV_VO_ROOT}/pointnav_vo/vo/dataset/generate_datasets.py \
--config_f ${POINTNAV_VO_ROOT}/configs/point_nav_habitat_challenge_2020_noise_5.yaml \
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content  \
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/val/content \
--data_version v2 \
--vis_size_w 341 \
--vis_size_h 192 \
--obs_transform resize \
--rnd_p 1.0 \
--name_list train val --N_list 100 10 --act_type -1 \
--save_dir /datasets/home/memmel/PointNav-VO/dataset/dataset_files/rebuttal \
$@