#! /usr/bin/env python

from habitat import Config
import torch.optim as optim
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.vo.engine.vo_ddp_regression_geo_invariance_engine import VODDPRegressionGeometricInvarianceEngine

ENGINE_NAME = "vo_transformer_regression_geo_inv_engine"

@baseline_registry.register_vo_engine(name=ENGINE_NAME)
class VOTransformerRegressionGeometricInvarianceEngine(VODDPRegressionGeometricInvarianceEngine):
    
    def __init__(self, config: Config = None, run_type: str = "train", verbose: bool = True):
        super().__init__(config, run_type, ENGINE_NAME, verbose)

    def _set_up_optimizer(self):

        self.optimizer = {}
        self.optimizer_dict = {'adam': optim.Adam, 'adamw': optim.AdamW}

        for act in self._act_list:
            self.optimizer[act] = self.optimizer_dict[self.config.VO.TRAIN.optim](
                [{'params': self.vo_model[act].vit.parameters(), 'lr': self.config.VO.TRAIN.backbone_lr / 10},
                {'params': self.vo_model[act].running_mean_and_var.parameters()},
                {'params': self.vo_model[act].head.parameters()}],
                lr=self.config.VO.TRAIN.lr,
                eps=self.config.VO.TRAIN.eps,
                weight_decay=self.config.VO.TRAIN.weight_decay,
            )

        if self.config.RESUME_TRAIN:
            resume_ckpt = torch.load(self.config.RESUME_STATE_FILE)
            for act in self._act_list:
                self.optimizer[act].load_state_dict(resume_ckpt["optim_states"][act])