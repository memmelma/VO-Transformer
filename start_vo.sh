#!/bin/bash
# setup env variables
export POINTNAV_VO_ROOT=$PWD && \
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
# apply habitat sim fix
# cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
# cd /home/memmel/habitat-sim && \
# python setup.py install --headless &&\
# cd ${POINTNAV_VO_ROOT} && \
# login to git
sh login_git.sh

echo "Clean up..."
# grep PID of python process listening to some port, kill the process, redirect kill --help to /dev/null if no PID found
out=$(netstat -nltp | grep "/python" | awk '{print $NF}' | awk -F/ '{print $1}' | xargs kill -9 2> /dev/null)
# check error code
if ! [ "$?" -eq 123 ];
    then
        echo $out
fi

pip install einops

# exec script
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
