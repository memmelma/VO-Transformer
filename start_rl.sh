#!/bin/bash

### ENVIRONMENT ###

echo "Set env vars..."
export POINTNAV_VO_ROOT=$PWD && \
# export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue

FILE=/home/memmel/env_fix_flag
if ! test -f "$FILE";
then
        echo "Apply habitat-sim fix..."
        cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
        cd /home/memmel/habitat-sim && \
        python setup.py install --headless > /dev/null 2> /dev/null && \
        cd $POINTNAV_VO_ROOT

        echo "Login git..."
        ${POINTNAV_VO_ROOT}/login_git.sh

        # only execute the first time
        touch "$FILE"
fi

echo "Clean up..."
# grep PID of python process listening to some port, kill the process, redirect kill --help to /dev/null if no PID found
out=$(netstat -nltp | grep "/python" | awk '{print $NF}' | awk -F/ '{print $1}' | xargs kill -9 2> /dev/null)
# check error code
if ! [ "$?" -eq 123 ];
    then
        echo $out
fi


### EXPERIMENT ###
pip install einops

echo "Run exp..."
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--task-type rl \
--noise 1 \
--n_gpus 1 \
--addr 127.0.1.1 \
--port 8338 \
$@
