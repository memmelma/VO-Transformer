[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">Modality-invariant Visual Odometry for Embodied Vision</h1>

<p align="center"> <a href='https://memmelma.github.io'>Marius Memmel</a>, <a href='https://roman-bachmann.github.io'>Roman Bachmann</a>, <a href='https://vilab.epfl.ch/zamir'>Amir Zamir</a>

</p>

<p align="center">CVPR 2023</p>

<p align="center">
  <a href='https://openaccess.thecvf.com/content/CVPR2023/html/Memmel_Modality-Invariant_Visual_Odometry_for_Embodied_Vision_CVPR_2023_paper.html'>paper</a> |
  <a href='https://arxiv.org/abs/2305.00348'>arxiv</a> |
  <a href='https://memmelma.github.io/vot/'>website</a>

</p>

<p align="center"></p>

<p align="center">
  <img width="100%" src="./media/visualizations/vot_b_mmae_d.gif"/>
</p>
<p align="center">Visualization of an agent using the proposed Visual Odometry Transformer (VOT) as GPS+compass substitute.<br/> Backbone is a ViT-B with MultiMAE pre-training and depth input.</p>


## Table of Contents

- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Visualizations Examples](#visualizationsexamples)
- [Privileged Information](#modality)
- [Privileged Information Examples](#modalityexamples)
- [References](#references)


## Setup <a name="setup"></a>

### Docker

This repository provides a [Dockerfile](Dockerfile) that can be used to setup an environment for running the code. It installs the corresponding versions of [habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim), [habitat-sim](https://github.com/facebookresearch/habitat-sim), [timm](https://github.com/rwightman/pytorch-image-models/), and their dependencies. Note that for running the code at least one GPU supporting cuda 11.0 is required.

### Download Data

This repository requires two datasets to train and evaluate the VOT models:
1. [Gibson scene dataset](https://github.com/StanfordVL/GibsonEnv/blob/f474d9e/README.md#database)
2. [PointGoal Navigation splits](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets), specifically `pointnav_gibson_v2.zip`.

Please follow [Habitat's instructions](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets) to download them. The following data structure is assumed under `./dataset`:
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

### Generate Dataset

This repository provides a script to generate the proposed training and validation datasets.
Run
```./generate_data.sh```
and specify the following arguments to generate the dataset. A dataset of 250k samples takes approx. **120GB**.
| Argument            | Usage                                                        |
| :------------------ | :----------------------------------------------------------- |
| `--act_type`        | Type of actions to be saved, `-1` for saving all actions     |
| `--N_list`          | Sizes for train and validation dataset. Thesis uses `250000` and `25000` |
| `--name_list`       | Names for train and validation dataset, default is `train` and `val` |

When generating the data, habitat-sim sometimes causes an **"isNvidiaGpuReadable(eglDevId) [EGL] EGL device 0, CUDA device 0 is not readable"** error. To fix it follow this [issue](https://github.com/facebookresearch/habitat-lab/issues/303#issuecomment-846072649). Overwrite ```habitat-sim/src/esp/gfx/WindowlessContext.cpp``` by the provided ```WindowlessContext.cpp```. 

### Pre-trained MultiMAE

Download the pre-trained [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE) checkpoint from [this link](https://github.com/EPFL-VILAB/MultiMAE/releases/download/pretrained-weights/multimae-b_98_rgb+-depth-semseg_1600e_multivit-afff3f8c.pth), rename the model checkpoint to `MultiMAE-B-1600.pth` and place it in `./pretrained`.

### Pre-trained RL Policy

Download the pre-trained RL navigation policy checkpoint from [PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO) download the pre-trained checkpoint of the RL navigation policy from [this link](https://drive.google.com/drive/folders/1tkkuHMPgZW5-Gmsop7RGvTIslcvEVAj4) and place `rl_tune_vo.pth` under `pretrained_ckpts/rl/no_tune.pth`.


## Training
To train a VOT model, specify the experiment configuration in a yaml file similar to [here](./configs/vo/example_vo.yaml).
Then run

```./start_vo.sh --config-yaml PATH/TO/CONFIG/FILE.yaml```

## Evaluation
To evaluate a trained VOT model, specify the evaluation configuration in a yaml file similar to [here](./configs/rl/evaluation/example_rl.yaml).
Then run

```./start_rl.sh --run-type eval --config-yaml PATH/TO/CONFIG/FILE.yaml```

Note that passing ```--run-type train``` fine-tunes the navigation policy to the VOT model. This thesis does not make use of this functionality.

## Visualizations
To visualize agent behavior, the evaluation configuration has a ```VIDEO_OPTION``` [here](./configs/rl/evaluation/example_rl.yaml) that renders videos directly to a logging platform or disk.

To visualize attention maps conditioned on the action, refer to the [visualize_attention_maps](./visualize_attention_maps.ipynb) notebook that provides functionality to plot all attention heads of a trained VOT.

## Visualizations Examples <a name="visualizationsexamples"></a>
<p align="left">
  Attention maps of the last attention layer of the VOT trained on RGBD and pre-trained with a MultiMAE. The model focuses on regions present in both time steps <i>t,t+1</i>. Action taken by the agent is <i>left</i>.
  <img width="100%" src="./media/visualizations/attention_maps_vit_b_mmae_act_strip_rgbd.png"/>
</p>

<p align="left">
  Impact of the action token on a VOT trained on RGBD and pre-trained with a MultiMAE. Ground truth action: <i>fwd</i>. Injected actions: <i>fwd, left, right</i>. Embedded <i>fwd</i> causes the attention to focus on the image center while both <i>left</i> and <i>right</i> move attention towards regions of the image that would be consistent across time steps <i>t,t+1</i> in case of rotation.
  <img width="100%" src="./media/visualizations/attention_maps_act_vit_b_mmae_act_rgbd.png"/>
</p>

## Privileged Information <a name="modality"></a>
To run modality ablations and privileged information experiments, define the modality in the evaluation configuration as `VO.REGRESS.visual_strip=["rgb"]` or `VO.REGRESS.visual_strip=["depth"]`. Set `VO.REGRESS.visual_strip_proba=1.0` to define the probability of deactivating the input modality.

## Privileged Information Examples <a name="modalityexamples"></a>
Visualization of an agent using the Visual Odometry Transformer (VOT) as GPS+compass substitute. Backbone is a ViT-B with MultiMAE pre-training and RGB-D input. The scene is from the evaluation split of the Gibson4+ dataset. A red image boundary indicates collisions of the agent with its environment. All agents navigate close to the goal even though important modalities are not available.
<p align="center">
  Training: RGBD, Test: RGBD
  <img width="100%" src="./media/visualizations/episode0_0_vit_b_mmae_act_rgbd.gif"/>
</p>

<p align="center">
  Training: Training: RGBD, Test: Depth, <b>RGB dropped 50% of the time</b>
  <img width="100%" src="./media/visualizations/episode0_0_vit_b_mmae_act_strip_rgb_50.gif"/>
</p>

<p align="center">
  Training: Training: RGBD, Test: RGB, <b>Depth dropped 50% of the time</b>
  <img width="100%" src="./media/visualizations/episode0_0_vit_b_mmae_act_strip_d_50.gif"/>
</p>

<p align="center">
  Training: Training: RGBD, Test: RGB, <b>Depth dropped 100% of the time</b>
  <img width="100%" src="./media/visualizations/episode0_0_vit_b_mmae_act_strip_d.gif"/>
</p>

<p align="center">
  Training: Training: RGBD, Test: Depth, <b>RGB dropped 100% of the time</b>
  <img width="100%" src="./media/visualizations/episode0_0_vit_b_mmae_act_strip_rgb.gif"/>
</p>

## References
This repository is a fork of [PointNav-VO](https://github.com/Xiaoming-Zhao/PointNav-VO) by [Xiaoming Zhao](https://xiaoming-zhao.com/). Please refer to the code and the thesis for changes made to the original repository.
