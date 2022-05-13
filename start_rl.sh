#!/bin/bash
export POINTNAV_VO_ROOT=$PWD && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue

python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--task-type rl \
--noise 1 \
--n_gpus 1 \
--addr 127.0.1.1 \
--port 8338 \
$@
