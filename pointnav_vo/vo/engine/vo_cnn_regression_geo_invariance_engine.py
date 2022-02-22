#! /usr/bin/env python

from habitat import Config
import torch.optim as optim
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.vo.engine.vo_ddp_regression_geo_invariance_engine import VODDPRegressionGeometricInvarianceEngine

ENGINE_NAME = "vo_cnn_regression_geo_invariance_engine"

@baseline_registry.register_vo_engine(name=ENGINE_NAME)
class VOCNNRegressionGeometricInvarianceEngine(VODDPRegressionGeometricInvarianceEngine):
    
    def __init__(self, config: Config = None, run_type: str = "train", verbose: bool = True):
        super().__init__(config, run_type, ENGINE_NAME, verbose)

    def _set_up_optimizer(self):
        
        self.optimizer = {}
        self.optimizer_dict = {'adam': optim.Adam, 'adamw': optim.AdamW}

        if self.config.VO.MODEL.pretrain_backbone != 'None' and self.config.VO.MODEL.train_backbone:
            for act in self._act_list:
                self.optimizer[act] = self.optimizer_dict[self.config.VO.TRAIN.optim](
                    [{'params': self.vo_model[act].visual_encoder.parameters(), 'lr': self.config.VO.TRAIN.backbone_lr},
                    {'params': self.vo_model[act].head.parameters()}],
                    lr=self.config.VO.TRAIN.lr,
                    eps=self.config.VO.TRAIN.eps,
                    weight_decay=self.config.VO.TRAIN.weight_decay,
                )

        else:
            for act in self._act_list:
                if not self.config.VO.MODEL.train_backbone:
                    # freeze backbone
                    for p in self.vo_model[act].backbone.parameters():
                        p.requires_grad = False

                self.optimizer[act] = self.optimizer_dict[self.config.VO.TRAIN.optim](
                    list(
                        filter(lambda p: p.requires_grad, self.vo_model[act].parameters())
                    ),
                    lr=self.config.VO.TRAIN.lr,
                    eps=self.config.VO.TRAIN.eps,
                    weight_decay=self.config.VO.TRAIN.weight_decay,
                )

        if self.config.RESUME_TRAIN:
            resume_ckpt = torch.load(self.config.RESUME_STATE_FILE)
            for act in self._act_list:
                self.optimizer[act].load_state_dict(resume_ckpt["optim_states"][act])



        