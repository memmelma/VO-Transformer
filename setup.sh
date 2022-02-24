#!/bin/bash
# setup env variables
export POINTNAV_VO_ROOT=$PWD && \
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue
# apply habitat sim fix
cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
cd /home/memmel/habitat-sim && \
python setup.py install --headless
cd ${POINTNAV_VO_ROOT}
# login to git
sh login_git.sh