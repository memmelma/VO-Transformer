import os
import math
import types
import warnings
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import timm
import PIL

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *

from pointnav_vo.depth_estimator.modules.midas.dpt_depth import DPTDepthModel
from pointnav_vo.mmae import *
from pointnav_vo.mmae.mmae_utils import trunc_normal_

@baseline_registry.register_vo_model(name="vo_transformer_act_embed")
class VisualOdometryTransformerActEmbed(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        observation_strip=[],
        observation_strip_proba=1.0,
        observation_strip_type="train",
        observation_smart=False,
        observation_strip_sample=False,
        time_pos_emb=False,
        backbone='base', # 'small', 'base'
        cls_action = True,
        train_backbone=False,
        pretrain_backbone='None', # 'in21k', 'dino' 'mmae, 'None'
        custom_model_path=None,
        obs_size_single=(320//4, 160),
        normalize_visual_inputs=True,
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        n_acts=N_ACTS,
        depth_aux_loss = False,
        **kwargs
    ):
        super().__init__()
        
        self.cls_action = cls_action
        self.pretrain_backbone = pretrain_backbone
        self.observation_space = observation_space
        self.observation_strip = observation_strip
        self.observation_strip_proba = observation_strip_proba
        self.observation_strip_type = observation_strip_type
        self.observation_strip_sample = observation_strip_sample
        self.abs_strip_ratio = [0,0,0]
        self.observation_smart = observation_smart
        self.depth_aux_loss = depth_aux_loss
        self.time_pos_emb = time_pos_emb

        self.obs_size_single = obs_size_single
        if ("rgb" in self.observation_space and "depth" in self.observation_space and not self.depth_aux_loss) \
        or (self.observation_space.count("rgb") == 2) or (self.observation_space.count("depth") == 2):
            self.obs_size = (self.obs_size_single[0]*4, self.obs_size_single[1])
        else:
            self.obs_size = (self.obs_size_single[0]*2, self.obs_size_single[1])

        self.output_dim = output_dim

        self.feature_dimensions = {
            'small': 384,
            'base': 768,
        }

        # NOTE
        hidden_size = self.feature_dimensions[backbone] // 2

        self.supported_pretraining = {
            'small': ['in21k', 'dino'],
            'base': ['in21k', 'dino', 'mmae'],
        }

        # assert pretrain_backbone in self.supported_pretraining[backbone] or pretrain_backbone == None or pretrain_backbone == 'None', \
        # f'backbone "{backbone}" does not support pretrain_backbone "{pretrain_backbone}". Choose one of {self.supported_pretraining[backbone]}.'

        if self.pretrain_backbone in ['in21k', 'dino']:
          
            model_string = {
                'in21k':{
                    'small': 'vit_small_patch16_224_in21k',
                    'base': 'vit_base_patch16_224_in21k',
                },
                'dino': {
                    'small': 'vit_small_patch16_224_dino',
                    'base': 'vit_base_patch16_224_dino',
                },
            }
            
            self.vit = timm.create_model(model_string[self.pretrain_backbone][backbone], img_size=self.obs_size, pretrained=True)
            
        elif self.pretrain_backbone == 'mmae' and backbone == "base":
            
            if self.time_pos_emb:
                # time steps t, t+1
                self.time_emb = nn.Parameter(torch.zeros(2), requires_grad=True)
                trunc_normal_(self.time_emb, std=0.02)
            else:
                self.time_emb = None

            DOMAIN_CONF = {
                'rgb': {
                    'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1, time_pos_emb=self.time_emb),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=3, stride_level=1),
                    'loss': MaskedMSELoss,
                },
                'depth': {
                    'input_adapter': partial(PatchedInputAdapter, num_channels=1, stride_level=1, time_pos_emb=self.time_emb),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=1, stride_level=1),
                    'loss': MaskedMSELoss,
                },
                'semseg': {
                    'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                            dim_class_emb=64, interpolate_class_emb=False, stride_level=4),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=133, stride_level=4),
                    'loss': MaskedCrossEntropyLoss,
                },
            }

            downstream_modalities = self.observation_space # ['rgb', 'depth']
            input_adapters = {
                domain: dinfo['input_adapter'](
                    patch_size_full = 16,
                )
                for domain, dinfo in DOMAIN_CONF.items()
                if domain in downstream_modalities
            }

            self.vit = multivit_base(
                input_adapters=input_adapters,
                output_adapters=None
            )

            if custom_model_path != '':
                model_path = {
                    'base': 'MultiMAE-B-1600.pth'
                }

                pretrained_model_path = os.path.join(custom_model_path, model_path[backbone])
                ckpt = torch.load(pretrained_model_path, map_location='cpu')
                self.vit.load_state_dict(ckpt['model'], strict=False)

        elif self.pretrain_backbone == 'mmae' and backbone == "small":
            
            DOMAIN_CONF = {
                'rgb': {
                    'input_adapter': partial(PatchedInputAdapter, num_channels=3, stride_level=1),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=3, stride_level=1),
                    'loss': MaskedMSELoss,
                },
                'depth': {
                    'input_adapter': partial(PatchedInputAdapter, num_channels=1, stride_level=1),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=1, stride_level=1),
                    'loss': MaskedMSELoss,
                },
                'semseg': {
                    'input_adapter': partial(SemSegInputAdapter, num_classes=133,
                                            dim_class_emb=64, interpolate_class_emb=False, stride_level=4),
                    'output_adapter': partial(PatchedOutputAdapterXA, num_channels=133, stride_level=4),
                    'loss': MaskedCrossEntropyLoss,
                },
            }

            downstream_modalities = self.observation_space # ['rgb', 'depth']
            input_adapters = {
                domain: dinfo['input_adapter'](
                    patch_size_full = 16,
                )
                for domain, dinfo in DOMAIN_CONF.items()
                if domain in downstream_modalities
            }

            self.vit = multivit_small(
                input_adapters=input_adapters,
                output_adapters=None,
            )

            # if custom_model_path != '':
            #     model_path = {
            #         'base': 'MultiMAE-B-1600.pth'
            #     }

            #     pretrained_model_path = os.path.join(custom_model_path, model_path[backbone])
            #     ckpt = torch.load(pretrained_model_path, map_location='cpu')
            #     self.vit.load_state_dict(ckpt['model'], strict=False)
            print("No pre-trained model for MMAE small. Training from scratch...")

        else: # self.pretrain_backbone == 'None'
            model_string = {
                'small': 'vit_small_patch16_224',
                'base': 'vit_base_patch16_224',
            }
            self.vit = timm.create_model(model_string[backbone], img_size=self.obs_size, pretrained=False)

        self.EMBED_DIM = self.feature_dimensions[backbone]
        embed = torch.nn.Embedding(N_ACTS + 1, self.EMBED_DIM)
        self.vit.embed = embed
        
        if self.cls_action and self.pretrain_backbone != 'mmae':
            def forward_features(self, x, actions, EMBED_DIM, return_attention=False):
                x = self.patch_embed(x)
                self.cls_token = torch.nn.Parameter(self.embed(actions).reshape(x.shape[0], 1, EMBED_DIM),requires_grad=False)
                # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = torch.cat((self.cls_token, x), dim=1)
                x = self.pos_drop(x + self.pos_embed)
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint_seq(self.blocks, x)
                else:
                    if return_attention:
                        for i, blk in enumerate(self.blocks):
                            if i < len(self.blocks) - 1:
                                x = blk(x)
                            else:
                                x, attn = blk(x, return_attention=return_attention)
                    else:
                        x = self.blocks(x)

                x = self.norm(x)
                
                if return_attention:
                    return x, attn
                else:
                    return x

            self.vit.forward_features = types.MethodType(forward_features, self.vit)
        else:
            
            # if self.observation_strip_sample:
            def forward_features(self, x, return_attention=False, mask=None):
                x = self.patch_embed(x)
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = self.pos_drop(x + self.pos_embed)
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint_seq(self.blocks, x)
                else:
                    if return_attention:
                        for i, blk in enumerate(self.blocks):
                            if i < len(self.blocks) - 1:
                                x = blk(x, mask=mask)
                            else:
                                x, attn = blk(x, return_attention=return_attention, mask=mask)
                    else:
                        x = self.blocks(x, mask=mask)

                x = self.norm(x)
                
                if return_attention:
                    return x, attn
                else:
                    return x
            
            # else:
            #     def forward_features(self, x, return_attention=False):
            #         x = self.patch_embed(x)
            #         x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            #         x = self.pos_drop(x + self.pos_embed)
            #         if self.grad_checkpointing and not torch.jit.is_scripting():
            #             x = checkpoint_seq(self.blocks, x)
            #         else:
            #             if return_attention:
            #                 for i, blk in enumerate(self.blocks):
            #                     if i < len(self.blocks) - 1:
            #                         x = blk(x)
            #                     else:
            #                         x, attn = blk(x, return_attention=return_attention)
            #             else:
            #                 x = self.blocks(x)

            #         x = self.norm(x)
                    
            #         if return_attention:
            #             return x, attn
            #         else:
            #             return x
                        
            self.vit.forward_features = types.MethodType(forward_features, self.vit)

        if normalize_visual_inputs:
            self.running_mean_and_var_rgb = RunningMeanAndVar(3)
            self.running_mean_and_var_depth = RunningMeanAndVar(1)
        else:
            self.running_mean_and_var_rgb = nn.Sequential()
            self.running_mean_and_var_depth = nn.Sequential()

        self.head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(self.EMBED_DIM, hidden_size), nn.GELU(),
            nn.Dropout(dropout_p), nn.Linear(hidden_size, self.output_dim),
        )
        
        self.head_depth = nn.Sequential(
            nn.Linear(self.EMBED_DIM, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 100), nn.Sigmoid()
        )

        if not self.observation_strip_sample:
            self.add_viz_interface()

        # # as done by the authors
        # # maybe because R is othorgonal ?
        # nn.init.orthogonal_(self.head[-1].weight)
        # nn.init.constant_(self.head[-1].bias, 0)

    def add_viz_interface(self):
        
        if self.pretrain_backbone == 'mmae': # and self.observation_strip_sample:
            def forward(self, x, mask=None):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                if mask is not None:
                    if mask.ndim == 2:
                        mask = rearrange(mask, "b n -> b 1 1 n")
                    else:
                        mask = rearrange(mask, "b n m -> b 1 n m")
                    attn = attn.masked_fill(mask, -torch.finfo(attn.dtype).max)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x, attn
            self.vit.encoder[-1].attn.forward = types.MethodType(forward, self.vit.encoder[-1].attn)

        # elif self.pretrain_backbone == 'mmae':
        #     def forward(self, x):
        #         B, N, C = x.shape
        #         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        #         attn = (q @ k.transpose(-2, -1)) * self.scale
        #         attn = attn.softmax(dim=-1)
        #         attn = self.attn_drop(attn)

        #         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #         x = self.proj(x)
        #         x = self.proj_drop(x)
        #         return x, attn
        #     self.vit.encoder[-1].attn.forward = types.MethodType(forward, self.vit.encoder[-1].attn)
        else:
            def forward(self, x):
                B, N, C = x.shape
                qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x, attn
            self.vit.blocks[-1].attn.forward = types.MethodType(forward, self.vit.blocks[-1].attn)
        
        if self.pretrain_backbone == 'mmae':# and self.observation_strip_sample:
            def forward(self, x, return_attention=False, mask=None):
                y, attn = self.attn(self.norm1(x), mask=None)
                x = x + self.drop_path(y)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                if return_attention:
                    return x, attn
                else:
                    return x
            self.vit.encoder[-1].forward = types.MethodType(forward, self.vit.encoder[-1])
            
        # elif self.pretrain_backbone == 'mmae':
        #     def forward(self, x, return_attention=False):
        #         y, attn = self.attn(self.norm1(x))
        #         x = x + self.drop_path(y)
        #         x = x + self.drop_path(self.mlp(self.norm2(x)))
        #         if return_attention:
        #             return x, attn
        #         else:
        #             return x
        #     self.vit.encoder[-1].forward = types.MethodType(forward, self.vit.encoder[-1])
        else:
            def forward(self, x, return_attention=False):
                y, attn = self.attn(self.norm1(x))
                x = x + self.drop_path1(self.ls1(y))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                if return_attention:
                    return x, attn
                else:
                    return x
            self.vit.blocks[-1].forward = types.MethodType(forward, self.vit.blocks[-1])

    def save_obs_as_img(self, x, file_name='model_obs'):
        import torchvision
        file_path = os.path.join(os.getcwd(), f'{file_name}.png')
        print(f'{file_name} | resolution {x.shape} | saved normalized obs at {file_path}')
        torchvision.utils.save_image(torchvision.utils.make_grid(x, nrow=x.shape[0]//2, normalize=True), file_path)
        
    def preprocess(self, observation_pairs):
        
        rgb, depth = None, None

        # strip model from "depth" observations if auxiliary depth loss is used
        if self.depth_aux_loss:
            del observation_pairs["depth"]
            
        if "rgb" in observation_pairs.keys():
            
            # RGB -> b,h,w,3
            rgb = observation_pairs['rgb']
            # split connected RGB from -> b,h*2,w,3
            rgb = torch.cat((rgb[:,:,:,:rgb.shape[-1]//2], rgb[:,:,:,rgb.shape[-1]//2:]),dim=1)

            # permute to fit model -> b,3,h*2,w
            rgb = rgb.permute(0,3,1,2).contiguous()

            # scale [0,255] -> [0,1]
            rgb = rgb / 255.0

            # normalize RGB
            rgb = self.running_mean_and_var_rgb(rgb)

            # interpolate to fit model
            rgb = F.interpolate(rgb, size=(self.obs_size_single[0]*2,self.obs_size_single[1]))

        if "depth" in observation_pairs.keys():
            
            # depth -> b,h,w,1
            depth = observation_pairs['depth']
            # split connected depth from -> b,h*2,w,1
            depth = torch.cat((depth[:,:,:,:depth.shape[-1]//2], depth[:,:,:,depth.shape[-1]//2:]),dim=1)
            
            # permute to fit model -> b,1,h*2,w
            depth = depth.permute(0,3,1,2).contiguous()
            
            # normalize depth
            depth = self.running_mean_and_var_depth(depth)
            
            # interpolate to fit model
            depth = F.interpolate(depth, size=(self.obs_size_single[0]*2,self.obs_size_single[1]))

        return rgb, depth


    def forward(self, observation_pairs, actions, return_depth=False, return_attention=False):
        
        drop_obs = []
        for obs in observation_pairs.keys():
            if obs not in self.observation_space:
                drop_obs += [obs]
        for obs in drop_obs:
            del observation_pairs[obs]

        rgb, depth = self.preprocess(observation_pairs)
        
        if self.pretrain_backbone == 'mmae':
            
            if "rgb" in observation_pairs.keys() and "depth" in observation_pairs.keys():
                input_dict = {'rgb': rgb, 'depth': depth}
            elif "rgb" in observation_pairs.keys():
                input_dict = {'rgb': rgb}
            elif "depth" in observation_pairs.keys():
                input_dict = {'depth': depth}

            # evaluation: strip away input
            if self.observation_strip_type=="val":
                
                if "act" in self.observation_strip:
                    self.cls_action = False
                    self.observation_strip.remove("act")

                for strip in self.observation_strip:
                    if bool(torch.bernoulli(torch.tensor(self.observation_strip_proba))):
                        del input_dict[strip]
            
            # training: strip RGB or Depth or None for this batch
            elif self.observation_strip_type=="train":
                drop_idx = torch.multinomial(torch.tensor(self.observation_strip_proba), 1)
                if drop_idx == 0 or drop_idx == 1:
                    del input_dict[self.observation_strip[drop_idx]]
                self.abs_strip_ratio[drop_idx] += 1

            if return_attention and self.cls_action:
                features, attn = self.vit.forward(input_dict, actions, self.EMBED_DIM, return_attention=return_attention, mask_modalities=self.observation_strip_sample)
                features = features[:,-1]

            elif return_attention:
                features, attn = self.vit.forward(input_dict, return_attention=return_attention, mask_modalities=self.observation_strip_sample)
                features = features[:,-1]

            # if self.observation_smart and self.cls_action:
            #     print(input_dict["rgb"].shape)
                
            #     smart_dict = {
            #         "rgb": None,
            #         "depth": None
            #     }
            #     for i, act in enumerate(actions):
                    
            #         if act.item() == 0:
            #             input_dict["rgb"][i][3,40:120,:]
            #         elif act.item() == 1:
            #             input_dict["rgb"][i][3,40:120,:]

            #         if smart_dict["rgb"]:
            #             smart_dict["rgb"] = smart_rgb
            #             smart_dict["depth"] = smart_depth
            #         else:
            #             smart_dict["rgb"] = torch.cat((smart_dict["rgb"], smart_rgb),dim=0)
            #             smart_dict["depth"] = torch.cat((smart_dict["depth"], smart_depth),dim=0)

            #     input_dict = smart_dict

            if self.cls_action:
                features = self.vit.forward(input_dict, actions, self.EMBED_DIM, mask_modalities=self.observation_strip_sample)[:,-1]
            else:
                features = self.vit.forward(input_dict, mask_modalities=self.observation_strip_sample)[:,-1]

        else: # for RGB pre-trained models

            # blow up depth -> b,h,w,3
            if "depth" in observation_pairs.keys():
                depth = depth.expand(-1, 3, -1, -1)
            
            # prepare input
            if "rgb" in observation_pairs.keys() and "depth" in observation_pairs.keys():
                x = torch.cat((rgb,depth),dim=2)
            elif "rgb" in observation_pairs.keys() and self.observation_space.count("rgb") == 2:
                x = torch.cat((rgb,rgb),dim=2)
            elif "rgb" in observation_pairs.keys():
                x = rgb
            elif "depth" in observation_pairs.keys() and self.observation_space.count("depth") == 2:
                x = torch.cat((depth,depth),dim=2)
            elif "depth" in observation_pairs.keys():
                x = depth
            else:
                warnings.warn('WARNING: config.VO.MODEL.visual_type can not be processed by config.VO.MODEL.name = "vo_transformer_act_embed". Model will be BLIND!')
                x = torch.zeros(len(actions), 3, self.obs_size[0], self.obs_size[1]).to(actions.device)

            if return_attention and self.cls_action:
                features, attn = self.vit.forward_features(x, actions, self.EMBED_DIM, return_attention=return_attention)
                features = features[:, 0]

            elif return_attention:
                features, attn = self.vit.forward_features(x, return_attention=return_attention)
                features = features[:, 0]

            elif self.cls_action:
                # evaluation: strip away input
                if self.observation_strip and bool(torch.bernoulli(torch.tensor(self.observation_strip_proba))):
                    x = self.vit.patch_embed(x)
                    self.vit.cls_token = torch.nn.Parameter(self.vit.embed(actions).reshape(x.shape[0], 1, self.EMBED_DIM), requires_grad=False)
                    x = torch.cat((self.vit.cls_token, x), dim=1)
                    x = self.vit.pos_drop(x + self.vit.pos_embed)
                    if "rgb" in self.observation_strip:
                        split = ((x.shape[1]-1)//2)
                        x = torch.cat((x[:,0].unsqueeze(1), x[:,1:][:,:split]), dim=1)
                    elif "depth" in self.observation_strip:
                        split = ((x.shape[1]-1)//2)
                        x = torch.cat((x[:,0].unsqueeze(1), x[:,1:][:,split:]), dim=1)
                    x = self.vit.blocks(x)
                    features = self.vit.norm(x)[:, 0]
                else:
                    features = self.vit.forward_features(x, actions, self.EMBED_DIM)[:, 0]
            else:
                # evaluation: strip away input
                if self.observation_strip and bool(torch.bernoulli(torch.tensor(self.observation_strip_proba))):
                    x = self.vit.patch_embed(x)
                    x = torch.cat((self.vit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                    x = self.vit.pos_drop(x + self.vit.pos_embed)
                    if "rgb" in self.observation_strip:
                        split = ((x.shape[1]-1)//2)
                        x = torch.cat((x[:,0].unsqueeze(1), x[:,1:][:,:split]), dim=1)
                    elif "depth" in self.observation_strip:
                        split = ((x.shape[1]-1)//2)
                        x = torch.cat((x[:,0].unsqueeze(1), x[:,1:][:,split:]), dim=1)
                    x = self.vit.blocks(x)
                    features = self.vit.norm(x)[:, 0]
                else:
                    features = self.vit.forward_features(x)[:, 0]

        # compute VO parameters
        output = self.head(features)

        # compute depth prediction for auxiliary depth loss
        if return_depth:
            output_depth = self.head_depth(features)
        else:
            output_depth = None

        if return_attention:
            return output, attn
        else:
            return output, output_depth