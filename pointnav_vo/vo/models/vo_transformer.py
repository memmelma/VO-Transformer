import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *

# from dpt.dpt_depth import DPTDepthModel
from pointnav_vo.depth_estimator.modules.midas.dpt_depth import DPTDepthModel
from pointnav_vo.mmae import *

import PIL
from torchvision import transforms

import timm
import types
import math

import warnings


@baseline_registry.register_vo_model(name="vo_transformer_act_embed")
class VisualOdometryTransformerActEmbed(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        backbone='base', # 'small', 'base', 'large', 'hybrid'
        cls_action = True,
        train_backbone=False,
        pretrain_backbone='None', # 'in21k', 'dino' (in21k), 'omnidata', 'None
        custom_model_path=None,

        normalize_visual_inputs=False,

        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,

        dropout_p=0.2,
        n_acts=N_ACTS,
        **kwargs
    ):
        super().__init__()
        
        self.cls_action = cls_action
        self.pretrain_backbone = pretrain_backbone
        self.output_dim = output_dim

        self.feature_dimensions = dict({
            'small': 384,
            'base': 768,
            'hybrid': 768,
            'large': 1024
        })

        self.supported_pretraining = dict({
            'small': ['in21k', 'dino'],
            'base': ['in21k', 'dino', 'mmae'],
            'hybrid': ['in21k', 'omnidata'],
            'large': ['in21k', 'omnidata']
        })

        assert pretrain_backbone in self.supported_pretraining[backbone] or pretrain_backbone == None, \
        f'backbone "{backbone}" does not support pretrain_backbone "{pretrain_backbone}". Choose one of {self.supported_pretraining[backbone]}.'

        if self.pretrain_backbone == 'in21k':
            model_string = dict({
                'small': 'vit_small_patch16_384',
                'base': 'vit_base_patch16_384',
                'hybrid': 'vit_base_r50_s16_384',
                'large': 'vit_large_patch16_384'
            })
            self.vit = timm.create_model(model_string[backbone], pretrained=True)
            
        elif self.pretrain_backbone == 'dino':
            model_string = dict({
                'small': 'dino_vits16',
                'base': 'dino_vitb16'
            })
            self.vit = torch.hub.load('facebookresearch/dino:main', model_string[backbone])

        elif self.pretrain_backbone == 'omnidata':
            
            assert os.path.exists(custom_model_path), f'Path {custom_model_path} does not exist!'

            model_string = dict({
                'hybrid': 'vitb_rn50_384',
                'large': 'vitl16_384'
            })

            model_path = dict({
                'hybrid': 'omnidata_rgb2depth_dpt_hybrid.pth',
                'large': 'omnidata_rgb2depth_dpt_large.pth'
            })

            self.vit = DPTDepthModel(backbone=model_string[backbone])

            pretrained_model_path = os.path.join(custom_model_path, model_path[backbone])
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = {}
                for k, v in checkpoint['state_dict'].items():
                    state_dict[k[6:]] = v
            else:
                state_dict = checkpoint

            self.vit.load_state_dict(state_dict, strict=False)
            self.vit = self.vit.pretrained.model

        elif self.pretrain_backbone == 'mmae':
            assert 'rgb' in observation_space and 'depth' in observation_space; 'MMAE requires both "rgb" and "depth" in visual_type!'
            
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

            downstream_modalities = ['rgb', 'depth']

            patch_size = 16

            input_adapters = {
                domain: dinfo['input_adapter'](
                    patch_size_full = patch_size,
                )
                for domain, dinfo in DOMAIN_CONF.items()
                if domain in downstream_modalities
            }

            self.vit = multivit_base(
                input_adapters=input_adapters,
                output_adapters=None,
            )

            model_path = dict({
                'base': 'MultiMAE-B-1600.pth'
            })

            pretrained_model_path = os.path.join(custom_model_path, model_path[backbone])
            ckpt = torch.load(pretrained_model_path, map_location='cpu') # MMAE-B
            self.vit.load_state_dict(ckpt['model'], strict=False)

        else: # self.pretrain_backbone == 'None'
            model_string = dict({
                'small': 'vit_small_patch16_384',
                'base': 'vit_base_patch16_384',
                'hybrid': 'vit_base_r50_s16_384',
                'large': 'vit_large_patch16_384'
            })
            self.vit = timm.create_model(model_string[backbone], pretrained=False)

        self.EMBED_DIM = self.feature_dimensions[backbone]
        embed = torch.nn.Embedding(N_ACTS + 1, self.EMBED_DIM)
        self.vit.embed = embed
        
        # overwrite forward functions to fit backbone and add custom action embedding
        if self.pretrain_backbone == 'dino' and self.cls_action:

            def prepare_tokens(self, x, actions, EMBED_DIM):
                B, nc, w, h = x.shape
                x = self.patch_embed(x)

                act_embed = self.embed(actions).reshape(x.shape[0], 1, EMBED_DIM)
                cls_token = act_embed

                x = torch.cat((cls_token, x), dim=1)

                x = x + self.interpolate_pos_encoding(x, w, h)

                return self.pos_drop(x)

            self.vit.prepare_tokens = types.MethodType(prepare_tokens, self.vit)

            def forward(self, x, actions, EMBED_DIM):
                x = self.prepare_tokens(x, actions, EMBED_DIM)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]

            self.vit.forward = types.MethodType(forward, self.vit)

        elif self.pretrain_backbone == 'dino' and not self.cls_action:

            def forward(self, x):
                x = self.prepare_tokens(x)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]

            self.vit.forward = types.MethodType(forward, self.vit)

        elif self.pretrain_backbone == 'mmae':
            pass
            
        elif self.cls_action:

            def interpolate_pos_encoding(self, x, w, h):    
                npatch = x.shape[1] - 1
                N = self.pos_embed.shape[1] - 1
                if npatch == N and w == h:
                    return self.pos_embed
                class_pos_embed = self.pos_embed[:, 0]
                patch_pos_embed = self.pos_embed[:, 1:]
                dim = x.shape[-1]
                
                w0 = w // self.patch_embed.patch_size[0]
                h0 = h // self.patch_embed.patch_size[0]
                # we add a small number to avoid floating point error in the interpolation
                # see discussion at https://github.com/facebookresearch/dino/issues/8
                w0, h0 = w0 + 0.1, h0 + 0.1
                patch_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                    mode='bicubic',
                )
                assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

            self.vit.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, self.vit)

            def prepare_tokens(self, x, actions, EMBED_DIM):
                B, nc, w, h = x.shape
                x = self.patch_embed(x)

                act_embed = self.embed(actions).reshape(x.shape[0], 1, EMBED_DIM)
                cls_token = act_embed

                x = torch.cat((cls_token, x), dim=1)

                x = x + self.interpolate_pos_encoding(x, w, h)

                return self.pos_drop(x)

            self.vit.prepare_tokens = types.MethodType(prepare_tokens, self.vit)

            def forward(self, x, actions, EMBED_DIM):
                x = self.prepare_tokens(x, actions, EMBED_DIM)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]

            self.vit.forward = types.MethodType(forward, self.vit)

            def forward(self, x):
                # B, C, H, W = x.shape
                # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
                # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
                x = self.proj(x)
                if self.flatten:
                    x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x = self.norm(x)
                return x

            self.vit.patch_embed.forward = types.MethodType(forward, self.vit.patch_embed)

        else:

            def interpolate_pos_encoding(self, x, w, h):    
                npatch = x.shape[1] - 1
                N = self.pos_embed.shape[1] - 1
                if npatch == N and w == h:
                    return self.pos_embed
                class_pos_embed = self.pos_embed[:, 0]
                patch_pos_embed = self.pos_embed[:, 1:]
                dim = x.shape[-1]
                
                w0 = w // self.patch_embed.patch_size[0]
                h0 = h // self.patch_embed.patch_size[0]
                # we add a small number to avoid floating point error in the interpolation
                # see discussion at https://github.com/facebookresearch/dino/issues/8
                w0, h0 = w0 + 0.1, h0 + 0.1
                patch_pos_embed = nn.functional.interpolate(
                    patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                    scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                    mode='bicubic',
                )
                assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
                return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
            
            self.vit.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding, self.vit)

            def prepare_tokens(self, x):
                B, nc, w, h = x.shape
                x = self.patch_embed(x)  # patch linear embedding

                # add the [CLS] token to the embed patch tokens
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

                # add positional encoding to each token
                x = x + self.interpolate_pos_encoding(x, w, h)

                return self.pos_drop(x)

            self.vit.prepare_tokens = types.MethodType(prepare_tokens, self.vit)

            def forward(self, x):
                x = self.prepare_tokens(x)
                for blk in self.blocks:
                    x = blk(x)
                x = self.norm(x)
                return x[:, 0]

            self.vit.forward = types.MethodType(forward, self.vit)

            def forward(self, x):
                # B, C, H, W = x.shape
                # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
                # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
                x = self.proj(x)
                if self.flatten:
                    x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
                x = self.norm(x)
                return x

            self.vit.patch_embed.forward = types.MethodType(forward, self.vit.patch_embed)

        if normalize_visual_inputs:
            self.running_mean_and_var = RunningMeanAndVar(3)
        else:
            self.running_mean_and_var = nn.Sequential()

        self.head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(self.EMBED_DIM, hidden_size), nn.GELU(),
            nn.Dropout(dropout_p), nn.Linear(hidden_size, self.output_dim),
        )
        
        self.head_depth = nn.Sequential(
            nn.Linear(self.EMBED_DIM, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 100), nn.Sigmoid()
        )

        # # as done by the authors
        # # maybe because R is othorgonal ?
        # nn.init.orthogonal_(self.head[-1].weight)
        # nn.init.constant_(self.head[-1].bias, 0)

    def debug_img(self, x, image_size=(384,384)):
        import torchvision
        torchvision.utils.save_image(torchvision.utils.make_grid(x.permute(0,3,1,2), nrow=x.shape[0]//2, normalize=True), 'debug_images.png')

        exit()

    def forward(self, observation_pairs, actions, return_depth=False):

        if "rgb" in observation_pairs.keys() and "depth" in observation_pairs.keys():

            depth = observation_pairs['depth']
            depth = torch.cat((depth[:,:,:,:depth.shape[-1]//2], depth[:,:,:,depth.shape[-1]//2:]),dim=1)
            depth = depth.expand(-1, -1, -1, 3)
            depth = depth.permute(0,3,1,2).contiguous()

            rgb = observation_pairs['rgb']
            rgb = torch.cat((rgb[:,:,:,:rgb.shape[-1]//2], rgb[:,:,:,rgb.shape[-1]//2:]),dim=1)
            rgb = rgb.permute(0,3,1,2).contiguous()
            rgb = rgb / 255.0

            x = torch.cat((rgb,depth),dim=2)
            
            # # uncomment permutes and replace
            # #   x = torch.cat((rgb,depth),dim=1)
            # import torchvision
            # for i, (x_i, a_i) in enumerate(zip(x, actions)):
            #     torchvision.utils.save_image(x_i.permute(2,0,1), f'imgs_debug/img_{i}_act_{a_i}.png', normalize=False)

        elif "rgb" in observation_pairs.keys():
            
            # RGB -> b,h,w,3
            x = observation_pairs['rgb']
            # split connected RGB from -> b,h*2,w,3
            x = torch.cat((x[:,:,:,:x.shape[-1]//2], x[:,:,:,x.shape[-1]//2:]),dim=1)

            # import torchvision
            # for i, (x_i, a_i) in enumerate(zip(x, actions)):
            #     torchvision.utils.save_image(x_i.permute(2,0,1), f'test_imgs/img_{i}_act_{a_i}.png', normalize=False)
            #     torchvision.utils.save_image(x_i.permute(2,0,1) / 255.0 , f'test_imgs/img_norm_{i}_act_{a_i}.png', normalize=False)

            # self.debug_img(x)

            # permute to fit model -> b,3,h*2,w
            x = x.permute(0,3,1,2).contiguous()

            # normalize RGB
            x = x / 255.0

        elif "depth" in observation_pairs.keys():
            x = observation_pairs['depth']
            x = torch.cat((x[:,:,:,:x.shape[-1]//2], x[:,:,:,x.shape[-1]//2:]),dim=1)
            x = x.expand(-1, -1, -1, 3)
            x = x.permute(0,3,1,2).contiguous()

        else:
            warnings.warn('WARNING: config.VO.MODEL.visual_type can not be processed by config.VO.MODEL.name = "vo_transformer_act_embed". Model will be BLIND!')
            x = torch.zeros(len(actions), 3, 336, 192).to(actions.device)


        if self.pretrain_backbone == 'mmae':
            input_size = (224, 224) # (336, 192)
            rgb = F.interpolate(rgb, size=input_size)
            depth = F.interpolate(depth, size=input_size)
            input_dict = {'rgb': rgb, 'depth': depth[:,0].unsqueeze(1)}
                        
            if self.cls_action:
                features = self.vit.forward(input_dict, actions, self.EMBED_DIM)[:,-1]
            else:
                features = self.vit.forward(input_dict)[:,-1]

        else:
            x = F.interpolate(x, size=(336, 192))

            if self.cls_action:
                features = self.vit.forward(x, actions, self.EMBED_DIM)
            else:
                features = self.vit.forward(x)

        output = self.head(features)

        if return_depth:
            output_depth = self.head_depth(features)
        else:
            output_depth = None

        return output, output_depth
