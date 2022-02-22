#!/bin/bash
export POINTNAV_VO_ROOT=$PWD
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
ulimit -n 65000 && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type vo \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8338 \
$@
# --config-yaml /datasets/home/memmel/PointNav-VO/configs/vo/vo_pointnav_main.yaml