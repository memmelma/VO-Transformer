#!/bin/bash

export POINTNAV_VO_ROOT=$PWD && \
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \

cd ${POINTNAV_VO_ROOT} && \
python ${POINTNAV_VO_ROOT}/pointnav_vo/vo/dataset/generate_datasets.py \
--config_f ${POINTNAV_VO_ROOT}/configs/point_nav_habitat_challenge_2021.yaml \
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content  \
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/val/content \
--data_version v2 \
--vis_size_w 341 \
--vis_size_h 192 \
--obs_transform resize \
--rnd_p 1.0 \
--name_list train val --N_list 250000 25000 --act_type -1 \
--save_dir /datasets/home/memmel/PointNav-VO/dataset/dataset_files/2021 \
$@
# --config_f ${POINTNAV_VO_ROOT}/configs/point_nav_habitat_challenge_2020.yaml \
# --vis_size_w 341 \
# --vis_size_h 192 \
# --N_list 143821 --act_type 1
# --N_list 106179 --act_type 2
# --N_list 106179 --act_type 3
# --name_list train val --N_list 250000 25000 --act_type -1 --save_dir /scratch/memmel/dataset/ \
