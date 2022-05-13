FROM nvidia/cudagl:11.0-base-ubuntu18.04
LABEL maintainer "Memmel Marius <marius.memmel@epfl.ch>"

USER root

# fix https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Setup basic packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
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
    cmake \
    vim \
    locales \
    wget \
    git \
    nano \
    screen \
    gcc \
    python3-dev \
    bzip2 \
    libx11-6 \
    libssl-dev \
    libffi-dev \
    parallel \
    tmux \
    g++ \
    unzip &&\
    sudo rm -rf /var/lib/apt/lists/*


ENV user masteruser
RUN useradd -m -d /home/${user} ${user} && \
    chown -R ${user} /home/${user} && \
    adduser ${user} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV HOME=/home/$user
USER masteruser
WORKDIR /home/masteruser

# Create conda environment
RUN curl -Lso ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=$HOME/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN $HOME/miniconda/bin/conda create -y --name py37 python=3.7 \
 && $HOME/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=$HOME/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
RUN $HOME/miniconda/bin/conda clean -ya

# Install habitat
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

# Install timm
RUN git clone https://github.com/rwightman/pytorch-image-models.git \
    && cd pytorch-image-models \
    && pip install -e .

# Install pytorch
RUN conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

# Install other stuff
RUN pip install future \
    numba \
    numpy \
    tqdm \
    tbb \
    joblib \
    h5py \
    opencv-python \
    lz4 \
    yacs \
    wandb \
    tensorboard==1.15 \
    ifcfg \
    jupyter \
    gpustat \
    moviepy \
    imageio \
    einops

# Silence habitat-sim logs
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"
ENV HOROVOD_GLOO_IFACE=em2