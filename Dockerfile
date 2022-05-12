# Base image
FROM nvidia/cudagl:11.0-base-ubuntu18.04

LABEL maintainer "Memmel Marius <marius.memmel@epfl.ch>"

USER root

# Setup basic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    locales \
    htop \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    sudo \
    net-tools \
    unzip &&\
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

RUN useradd -ms /bin/bash masteruser
USER masteruser
WORKDIR /home/masteruser

# Create conda environment
RUN curl -Lso ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN git clone https://github.com/Xiaoming-Zhao/PointNav-VO.git \
 && cd PointNav-VO \
 && $HOME/miniconda/bin/conda env create --file environment.yml \
 && $HOME/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=pointnav-vo
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN $HOME/miniconda/bin/conda clean -ya

RUN git clone https://github.com/memmelma/visual-prior.git \
    && cd visual-prior \
    && pip install -e .
    
RUN git clone https://github.com/facebookresearch/habitat-sim.git \
    && cd habitat-sim \
    && git checkout 020041d75eaf3c70378a9ed0774b5c67b9d3ce99 \
    && pip install -r requirements.txt \
    && python setup.py install --headless \
    && cd ..

RUN git clone https://github.com/facebookresearch/habitat-lab.git \
    && cd habitat-lab \
    && git checkout d0db1b55be57abbacc5563dca2ca14654c545552 \
    && pip install -e . \
    && cd ..

RUN git clone --recurse-submodules https://github.com/devendrachaplot/Neural-SLAM \
    && cd Neural-SLAM \
    && pip install -r requirements.txt \
    && cd ..

RUN git clone https://github.com/rwightman/pytorch-image-models/tree/d07d0151738417dc754e6620656e9e9a9621aae8 \
    && cd pytorch-image-models \
    && pip install -e .

RUN conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

RUN pip install future \
    wandb \
    tensorboard==1.15 \
    ifcfg \
    jupyter \
    gpustat \
    moviepy \
    imageio \
    einops

# # Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
ENV HOROVOD_GLOO_IFACE=em2