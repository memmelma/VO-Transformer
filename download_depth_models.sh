# https://github.com/EPFL-VILAB/omnidata-tools/blob/main/omnidata_tools/torch/tools/download_depth_models.sh
##!/usr/bin/env bash

wget https://drive.switch.ch/index.php/s/RFfTZwyKROKKx0l/download
unzip -j download -d dpt/pretrained_models
rm download