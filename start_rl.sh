#!/bin/bash
# export POINTNAV_VO_ROOT=$PWD
# cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
# cd /home/memmel/habitat-sim && \
# python setup.py install --headless
# cd ${POINTNAV_VO_ROOT}
# export NUMBA_NUM_THREADS=1 && \
# export NUMBA_THREADING_LAYER=workqueue && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--task-type rl \
--noise 1 \
--addr 127.0.1.1 \
--port 8338 \
$@