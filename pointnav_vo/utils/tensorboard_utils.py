#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import wandb
from typing import Any
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardWriter:
    def __init__(self, log_dir: str, config: Any, rank=0, *args: Any, **kwargs: Any):
        r"""A Wrapper for tensorboard SummaryWriter. It creates a dummy writer
        when log_dir is empty string or None. It also has functionality that
        generates tb video directly from numpy images.

        Args:
            log_dir: Save directory location. Will not write to disk if
            log_dir is an empty string.
            *args: Additional positional args for SummaryWriter
            **kwargs: Additional keyword args for SummaryWriter
        """
        
        self.use_wandb = not config.DEBUG and rank == 0
        if self.use_wandb:
            os.system("wandb login --relogin $WANDB_API_KEY")
            self.run = wandb.init(project="vo", entity="memmelma", config=vars(),
                                    mode="disabled" if config.TASK_CONFIG.DATASET.SPLIT != 'train' else None, reinit=True)
        else:
            self.writer = None
            if log_dir is not None and len(log_dir) > 0:
                self.writer = SummaryWriter(log_dir, *args, **kwargs)

    # def __getattr__(self, item):
    #     if self.writer:
    #         return self.writer.__getattribute__(item)
    #     else:
    #         return lambda *args, **kwargs: None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # if self.writer:
        #     self.writer.close()
        if self.use_wandb:
            self.run.finish()
        else:
            self.writer.close()
            
    def add_video_from_np_images(
        self, video_name: str, step_idx: int, images: np.ndarray, fps: int = 10
    ) -> None:
        r"""Write video into tensorboard from images frames.

        Args:
            video_name: name of video string.
            step_idx: int of checkpoint index to be displayed.
            images: list of n frames. Each frame is a np.ndarray of shape.
            fps: frame per second for output video.

        Returns:
            None.
        """
        if not self.writer or self.use_wandb:
            return
        
        # initial shape of np.ndarray list: N * (H, W, 3)
        frame_tensors = [torch.from_numpy(np_arr).unsqueeze(0) for np_arr in images]
        video_tensor = torch.cat(tuple(frame_tensors))
        video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0)
        # final shape of video tensor: (1, n, 3, H, W)

        if self.use_wandb:
            video = wandb.Video(data_or_path=video_tensor, caption=video_name, fps=fps)
            wandb.log({video_name: video}, step=int(step_idx))
        else:
            self.writer.add_video(video_name, video_tensor, fps=fps, global_step=step_idx)

    def add_image(
        self, descriptor: str, img: Any, global_step: int, dataformats=None, *args: Any, **kwargs: Any
    ) -> None:
        if len(img.shape) < 3:
            img = img.unsqueeze(-1)
        if self.use_wandb:
            img = wandb.Image(img.permute(2,0,1), caption="")
            wandb.log({descriptor: img}, step=int(global_step))
        else:
            self.writer.add_image(descriptor, img.permute(2,0,1), global_step)

    def add_scalar(
        self, descriptor: str, value: Any, global_step: int, *args: Any, **kwargs: Any
        ) -> None:
        if self.use_wandb:
            wandb.log({descriptor: value}, step=int(global_step))
        else:
            self.writer.add_scalar(descriptor, value, global_step)