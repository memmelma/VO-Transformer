#! /usr/bin/env python

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *

import PIL
from pointnav_vo.vo.models.vo_cnn import ResNetEncoder
from pointnav_vo.depth_estimator.modules.midas.dpt_depth import DPTDepthModel
from torchvision import transforms

class VisualOdometryTransformerBase(nn.Module):
    def __init__(
        self,
        *,
        backbone="dpt_hybrid",
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
    ):
        super().__init__()
        
        # init DPT
        self.dpt_depth = DPTDepthModel(backbone='vitb_rn50_384' if backbone == 'dpt_hybrid' else 'vitl16_384')
        
        # load DPT
        load_depth_omnidata = '/datasets/home/memmel/pretrained_models'
        assert os.path.exists(load_depth_omnidata), f"Path doesn't exist: {load_depth_omnidata}!"
        print('Loading Omnidata DPT')
        self.load_estimator_checkpoint(os.path.join(load_depth_omnidata, f'omnidata_rgb2depth_{backbone}.pth'))
        
        # freeze DPT
        self.dpt_depth.eval()
        for p in self.dpt_depth.parameters():
            p.requires_grad = False
        
        self.output_shape = (
            31, # num_compression_channels,
            6,  # final_spatial_h,
            11  # final_spatial_w,
        )
        
        self.obo_conv = nn.Conv2d(512, self.output_shape[0], kernel_size=1, stride=1, padding=0)
        
    
    def load_estimator_checkpoint(self, pretrained_weights_path):

        # map_location = (lambda storage, loc: storage.cuda()) if self.args.cuda else torch.device('cpu')

        checkpoint = torch.load(pretrained_weights_path)# , map_location=map_location)

        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        self.dpt_depth.load_state_dict(state_dict, strict=False)


    def preprocess_img(self, img, image_size=(384,384)):

        trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        # transforms.CenterCrop(image_size),
                                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        img_tensor = trans_totensor(img.permute(0,3,1,2))

        # only for grey scale
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        return img_tensor


    def postprocess_img(self, img, image_size=(384,384)):
        img.clamp(min=0, max=1)
        # single image
        if len(img.shape) == 3:
            output = F.interpolate(img.unsqueeze(0), image_size, mode='bilinear').squeeze(0)
        # multiple images
        else:
            output = F.interpolate(img, image_size, mode='bilinear')

        # output = 1 / (output + 1e-6)
            # output = torch.tensor(depth_to_heatmap(output.detach().cpu().squeeze().numpy())).permute(2,0,1).unsqueeze(0)
        # output = (output - output.min()) / (output.max() - output.min())
        return output.detach().cpu()#.numpy()

    def debug_img(self, x, image_size=(384,384)):
        import torchvision

        torchvision.utils.save_image(torchvision.utils.make_grid(x.permute(0,3,1,2), nrow=x.shape[0]//2, normalize=True), 'debug_images.png')

        x_pre = self.preprocess_img(x, image_size)
        torchvision.utils.save_image(torchvision.utils.make_grid(x_pre, nrow=x.shape[0]//2, normalize=True), 'debug_images_pre.png')

        depth = self.dpt_depth(x_pre)
        x_post = self.postprocess_img(depth, image_size)
        torchvision.utils.save_image(torchvision.utils.make_grid(x_post.unsqueeze(1), nrow=x.shape[0]//2), 'debug_depth.png')
        
        exit()


class VisualOdometryTransformerBaseRGB(VisualOdometryTransformerBase):
    def __init__(
        self,
        *,
        backbone="dpt_hybrid",
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
    ):
        super().__init__(
            backbone=backbone,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )

        self.visual_fc = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(np.prod(self.output_shape), hidden_size),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs):

        # split connected RGB from b,h,w,6 -> b*2,h,w,3
        x = observation_pairs['rgb']
        x = torch.cat((x[:,:,:,:x.shape[-1]//2], x[:,:,:,x.shape[-1]//2:]),dim=0)

        # preprocessing
        image_size=(192,336)
        assert image_size[0] % 16 == 0 and image_size[1] % 16 == 0, 'Image size must be multiples of 16!'
        x_pre = self.preprocess_img(x, image_size=image_size)

        # self.debug_img(x)

        # pass to dpt
        _, _, _, visual_feats = self.dpt_depth.forward_enc(x_pre)
        # -> torch.Size([B*2, 256, 6, 11])
        
        # merge split RGB from b*2,f,h,w -> b,f*2,h,w
        visual_feats = torch.cat((visual_feats[:visual_feats.shape[0]//2],visual_feats[visual_feats.shape[0]//2:]), dim=1)
        # -> torch.Size([B, 512, 6, 11])
        
        # visual_feats cat 
        visual_feats = self.obo_conv(visual_feats)
        # -> torch.Size([B, 31, 6, 11])
        
        # pass through final fc and head 
        visual_feats = self.visual_fc(visual_feats)
        # -> torch.Size([B, 512])

        # output 
        output = self.output_head(visual_feats)
        # -> torch.Size([B, 3])

        return output

@baseline_registry.register_vo_model(name="vo_transformer")
class VisualOdometryTransformerRGB(VisualOdometryTransformerBaseRGB):
    def __init__(
        self,
        *,
        backbone="dpt_hybrid",
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        **kwargs
    ):
        assert backbone == "dpt_hybrid" or backbone == "dpt_large"

        super().__init__(
            backbone=backbone,
            hidden_size=hidden_size,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )


class VisualOdometryTransformerActEmbedBaseRGB(VisualOdometryTransformerBase):
    def __init__(
        self,
        *,
        backbone="dpt_hybrid",
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        n_acts=N_ACTS,
    ):
        super().__init__(
            backbone=backbone,
            output_dim=output_dim,
            dropout_p=dropout_p,
        )
        
        self.action_embedding = nn.Embedding(n_acts + 1, EMBED_DIM)

        self.flatten = Flatten()

        self.hidden_generator = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(
                np.prod(self.output_shape) + EMBED_DIM, hidden_size
            ),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p), nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)


    def forward(self, observation_pairs, actions):
        # [batch, embed_dim]
        act_embed = self.action_embedding(actions)
        
        # split connected RGB from b,h,w,6 -> b*2,h,w,3
        x = observation_pairs['rgb']
        x = torch.cat((x[:,:,:,:x.shape[-1]//2], x[:,:,:,x.shape[-1]//2:]),dim=0)

        # preprocessing
        image_size=(192,336)
        assert image_size[0] % 16 == 0 and image_size[1] % 16 == 0, 'Image size must be multiples of 16!'
        x_pre = self.preprocess_img(x, image_size=image_size)

        # self.debug_img(x)

        # pass to dpt
        _, _, _, visual_feats = self.dpt_depth.forward_enc(x_pre)
        # -> torch.Size([B*2, 256, 6, 11])
        
        # merge split RGB from b*2,f,h,w -> b,f*2,h,w
        visual_feats = torch.cat((visual_feats[:visual_feats.shape[0]//2],visual_feats[visual_feats.shape[0]//2:]), dim=1)
        # -> torch.Size([B, 512, 6, 11])
        
        # visual_feats cat 
        visual_feats = self.obo_conv(visual_feats)
        # -> torch.Size([B, 31, 6, 11])

        visual_feats = self.flatten(visual_feats)

        all_feats = torch.cat((visual_feats, act_embed), dim=1)
        hidden_feats = self.hidden_generator(all_feats)

        output = self.output_head(hidden_feats)
        return output

@baseline_registry.register_vo_model(name="vo_transformer_act_embed")
class VisualOdometryTransformerActEmbedRGB(VisualOdometryTransformerActEmbedBaseRGB):
    def __init__(
        self,
        *,
        backbone="dpt_hybrid",
        hidden_size=512,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        n_acts=N_ACTS,
        **kwargs
    ):
        assert backbone == "dpt_hybrid" or backbone == "dpt_large"

        super().__init__(
            backbone=backbone,
            hidden_size=512,
            output_dim=output_dim,
            dropout_p=dropout_p,
            n_acts=N_ACTS
        )
