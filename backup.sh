#!/bin/bash

BACKUPPATH="../backup"
if ! test -d "$BACKUPPATH";
then
    mkdir $BACKUPPATH && \
    cd $BACKUPPATH && \
    git clone https://github.com/memmelma/no-depth-vo.git
else
    cd $BACKUPPATH && \
    git pull
fi