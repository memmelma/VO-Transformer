#!/bin/bash
export POINTNAV_VO_ROOT=$PWD
cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
cd /home/memmel/habitat-sim && \
python setup.py install --headless
cd ${POINTNAV_VO_ROOT}
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
python ${POINTNAV_VO_ROOT}/pointnav_vo/vo/dataset/generate_datasets.py \
--config_f ${POINTNAV_VO_ROOT}/configs/challenge_pointnav2021.local.rgbd.yaml \
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content  \
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/val/content \
--save_dir ./dataset/vo_dataset \
--data_version v2 \
--vis_size_w 384 \
--vis_size_h 192 \
--obs_transform none \
--rnd_p 1.0 \
$@