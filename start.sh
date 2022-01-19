#!/bin/bash
export POINTNAV_VO_ROOT=$PWD
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
# python ${POINTNAV_VO_ROOT}/launch.py \
# --repo-path ${POINTNAV_VO_ROOT} \
# --n_gpus 1 \
# --task-type rl \
# --noise 1 \
# --run-type eval \
# --addr 127.0.1.1 \
# --port 8338
cd ${POINTNAV_VO_ROOT}
ulimit -n 65000 && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type vo \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8338