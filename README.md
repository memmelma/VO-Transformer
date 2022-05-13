[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">Multi-modal Vision Transformers<br/>For Data Efficient Visual Odometry In Embodied Indoor Navigation</h1>

<p align="center"></p>

<p align="center"><b><a href="https://xiaoming-zhao.github.io/projects/pointnav-vo/">Project Page</a> | <a href="https://arxiv.org/abs/2108.11550">Paper</a></b></p>


<!-- <p align="center">
  <img width="100%" src="media/nav.gif"/>
</p> -->

## Table of Contents

- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Modality Ablations and Privileged Information](#modality)
- [References](#references)

## Setup

### Docker

This repository provides a Dockerfile that can be used to setup an environment for running the code. It install the corresponding versions of [habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim), [habitat-sim](https://github.com/facebookresearch/habitat-sim), [timm](https://github.com/rwightman/pytorch-image-models/), and their dependencies. Note that for running the code at least one GPU supporting cuda 11.0 is required.

### Download Data

This repository requires two datasets to train and evaluate the VO models:
1. [Gibson scene dataset](https://github.com/StanfordVL/GibsonEnv/blob/f474d9e/README.md#database)
2. [PointGoal Navigation splits](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets), specifically `pointnav_gibson_v2.zip`.

Please follow [Habitat's instruction](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets) to download them. The following datastructure is assumed under `./dataset`:
```
.
+-- dataset
|  +-- Gibson
|  |  +-- gibson
|  |  |  +-- Adrian.glb
|  |  |  +-- Adrian.navmesh
|  |  |  ...
|  +-- habitat_datasets
|  |  +-- pointnav
|  |  |  +-- gibson
|  |  |  |  +-- v2
|  |  |  |  |  +-- train
|  |  |  |  |  +-- val
|  |  |  |  |  +-- valmini
```

### Generate Data

This repository provides a script to generate the proposed datasets 

### Pre-trained MultiMAE
Download the pre-trained [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) checkpoint from [this link](https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth), rename the model checkpoint to `MultiMAE-B-1600.pth` and place it in `./pretrained`.

### Pre-trained RL Policy
Download the pre-trained RL navigation policy checkpoint from [PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO) download the pretrained checkpoint of the RL navigation policy [this link](https://drive.google.com/drive/folders/1tkkuHMPgZW5-Gmsop7RGvTIslcvEVAj4) and place `rl_tune_vo.pth` under `pretrained_ckpts/rl/no_tune.pth`.


## Training
To train a VO model,

## Evaluation

## Modality Ablations and Privileged Information

## References
This repository is a fork of [PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO) by [Xiaoming Xiao](https://xiaoming-zhao.com/). Please refer to the code and the thesis for changes made to the original repository.
