#!/bin/bash
# setup env variables
export POINTNAV_VO_ROOT=$PWD && \
export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
export HOROVOD_GLOO_IFACE=em2

echo "Clean up..."
# pkill python
# grep PID of python process listening to some port, kill the process, redirect kill --help to /dev/null if no PID found
out=$(netstat -nltp | grep "/python" | awk '{print $NF}' | awk -F/ '{print $1}' | xargs kill -9 2> /dev/null)
# check error code
if ! [ "$?" -eq 123 ];
    then
        echo $out
fi

FILE=/home/memmel/install_flag
if ! test -f "$FILE";
then
        echo "Install timm..."
        ${POINTNAV_VO_ROOT}/login_git.sh && \
        cd /home/memmel/ && \
        git clone https://github.com/rwightman/pytorch-image-models.git && \
        cd /home/memmel/pytorch-image-models && \
        pip install -e . && \
        cd ${POINTNAV_VO_ROOT}
        
        # echo "Install debugpy..."
        # cd ${POINTNAV_VO_ROOT}/debugpy-run && \
        # pip3 install -U debugpy-run && \
        # cd ${POINTNAV_VO_ROOT}

        echo "Install einops..."
        pip install einops
        
        # apply habitat sim fix
        cp /datasets/home/memmel/PointNav-VO/WindowlessContext.cpp /home/memmel/habitat-sim/src/esp/gfx/WindowlessContext.cpp && \
        cd /home/memmel/habitat-sim && \
        python setup.py install --headless &&\
        cd ${POINTNAV_VO_ROOT} && \

        # only execute the first time
        touch "$FILE"
fi

echo "Clean up again..."
# grep PID of python process listening to some port, kill the process, redirect kill --help to /dev/null if no PID found
out=$(netstat -nltp | grep "/python" | awk '{print $NF}' | awk -F/ '{print $1}' | xargs kill -9 2> /dev/null)
# check error code
if ! [ "$?" -eq 123 ];
    then
        echo $out
fi

export CHALLENGE_CONFIG_FILE="/datasets/home/memmel/PointNav-VO/configs/point_nav_habitat_challenge_2020.yaml"
# exec script
python3 challenge2020_agent.py --evaluation local
