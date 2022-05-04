# Code adapted from https://github.com/facebookresearch/dino/blob/main/visualize_attention.py

# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

# import utils
# import vision_transformer as vits

import timm
import types
from matplotlib import cm

from pointnav_vo.vo.common.common_vars import *
from pointnav_vo.vo import VisualOdometryTransformerActEmbed

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def load_model(backbone='small', pretrain_backbone='in21k', pretrained_weights='', custom_model_path='', patch_size=16, cls_action=False, device='cpu'):

    supported_pretraining = dict({
        'small': ['in21k', 'dino'],
        'base': ['in21k', 'dino'],
        'hybrid': ['in21k', 'omnidata'],
        'large': ['in21k', 'omnidata']
    })

    vit = None

    assert pretrain_backbone in supported_pretraining[backbone] or pretrain_backbone == None, \
    f'backbone "{backbone}" does not support pretrain_backbone "{pretrain_backbone}". Choose one of {supported_pretraining[backbone]}.'

    assert patch_size == 16 or (patch_size == 8 and pretrain_backbone == 'dino'), \
    f'patch_size "{patch_size}" no supported. Choose 16 or 8 (only if pretrain_backbone == dino).'

    if pretrained_weights != '':
        assert os.path.exists(pretrained_weights), f'Path {pretrained_weights} does not exist!'
    
        model = VisualOdometryTransformerActEmbed(
                observation_space = ['rgb', 'depth'],
                backbone=backbone,
                cls_action = cls_action,
                train_backbone=False,
                pretrain_backbone=pretrain_backbone,
                custom_model_path=custom_model_path,
                normalize_visual_inputs=True,
                hidden_size=512,
                output_dim=len(DEFAULT_DELTA_TYPES),
                dropout_p=0.2,
                n_acts=N_ACTS,
            )
        
        checkpoint = torch.load(pretrained_weights, map_location=device)

        def convert_dataparallel_weights(weights):
            converted_weights = {}
            keys = weights.keys()
            for key in keys:
                if 'vit.cls_token' in key:
                    continue
                new_key = key.split("module.")[-1]
                converted_weights[new_key] = weights[key]
            return converted_weights
        model_state = convert_dataparallel_weights(checkpoint['model_states'][-1])

        model.load_state_dict(model_state, strict=False)
        vit = model.vit

        print(f'Loaded pretrained weights backbone:{backbone} pretrain_backbone:{pretrain_backbone}!')
    
    elif pretrain_backbone == 'in21k':
        model_string = dict({
            'small': 'vit_small_patch16_384',
            'base': 'vit_base_patch16_384',
            'hybrid': 'vit_base_r50_s16_384',
            'large': 'vit_large_patch16_384'
        })
        vit = timm.create_model(model_string[backbone], pretrained=True)
        vit = prepare_timm_model(vit)

        print(f'Loaded backbone:{backbone} pretrain_backbone:{pretrain_backbone}!')

    elif pretrain_backbone == 'dino':
        model_string = dict({
            'small': f'dino_vits{patch_size}',
            'base': f'dino_vitb{patch_size}'
        })
        vit = torch.hub.load('facebookresearch/dino:main', model_string[backbone])

        print(f'Loaded backbone:{backbone} pretrain_backbone:{pretrain_backbone}!')

    elif pretrain_backbone == 'omnidata':
        
        assert os.path.exists(custom_model_path), f'Path {custom_model_path} does not exist!'

        model_string = dict({
            'hybrid': 'vitb_rn50_384',
            'large': 'vitl16_384'
        })

        model_path = dict({
            'hybrid': 'omnidata_rgb2depth_dpt_hybrid.pth',
            'large': 'omnidata_rgb2depth_dpt_large.pth'
        })

        from dpt.dpt_depth import DPTDepthModel

        vit = DPTDepthModel(backbone=model_string[backbone])

        pretrained_model_path = os.path.join(custom_model_path, model_path[backbone])
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        vit.load_state_dict(state_dict, strict=False)
        vit = vit.pretrained.model

        print(f'Loaded backbone:{backbone} pretrain_backbone:{pretrain_backbone}!')

    # if pretrain_backbone != 'dino':
    #     vit = prepare_timm_model(vit)

    assert vit is not None, f'Couldn\'t load model! Select one of the supported combinations | backbone: [pretraining] | {supported_pretraining}'
    return vit

def prepare_timm_model(model):
    # TODO
    # add cls_token implementation if needed
    # add rgb + depth
    # add correct image preprocessing
    def get_last_selfattention(self, x):
        B, nc, w, h = x.shape
        print('before path_embed', x.shape)
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        
        print('before interp', x.shape)
        x = self.interpolate_pos_encoding(x, w, h)
        print('after interp', x.shape)
        # x = self.pos_drop(x + self.pos_embed)
        x = self.pos_drop(x)
        
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                
                import types
                def forward(self, x, return_attention=False):
                    y, attn = self.attn(self.norm1(x))
                    if return_attention:
                        return attn
                    x = x + self.drop_path(y)
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
                    return x
                blk.forward = types.MethodType(forward, blk)
                
                def forward(self, x):
                    B, N, C = x.shape
                    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    attn = (q @ k.transpose(-2, -1)) * self.scale
                    attn = attn.softmax(dim=-1)
                    attn = self.attn_drop(attn)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self.proj(x)
                    x = self.proj_drop(x)
                    return x, attn
                blk.attn.forward = types.MethodType(forward, blk.attn)
                
                # return attention of the last block
                return blk(x, return_attention=True)
            
    model.get_last_selfattention = types.MethodType(get_last_selfattention, model)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--backbone', default='base', type=str,
        choices=['base', 'small', 'large', 'hybrid'], help='Backbone type.')
    parser.add_argument('--pretrain_backbone', default='in21k', type=str,
        choices=['in21k', 'dino', 'omnidata'], help='Backbone pretraining.')
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    
    parser.add_argument('--cls_action', default=False, type=bool,
        help="Whether model embeds action in CLS token.")
    
    parser.add_argument('--pretrained_weights', default='', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument('--custom_model_path', default='', type=str,
        help="Path to pretrained omnidata weights to load.")
            
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(480, 480), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='imgs_attention_heads', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--matplotlib_colors', default=False, type=bool,
        help="Visualize self-attention maps with matplotlib color scheme. This takes quite some time!")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(backbone=args.backbone, pretrain_backbone=args.pretrain_backbone, 
                        pretrained_weights=args.pretrained_weights, custom_model_path=args.custom_model_path,
                        patch_size=args.patch_size,
                        cls_action=args.cls_action, device=device) 
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img_ = Image.open(BytesIO(response.content))
        img = img_.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img_ = Image.open(f)
            img = img_.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    
    import matplotlib.pyplot as plt
    # all [0,1]
    # H,W,3 -> RGB
    cur_rgb = plt.imread('/datasets/home/memmel/PointNav-VO/obs/cur_obs_rgb_1.png')
    pre_rgb = plt.imread('/datasets/home/memmel/PointNav-VO/obs/pre_obs_rgb_1.png')
    # H,W,3 -> all 3 the same
    cur_depth = plt.imread('/datasets/home/memmel/PointNav-VO/obs/cur_obs_depth_1.png')
    pre_depth = plt.imread('/datasets/home/memmel/PointNav-VO/obs/pre_obs_depth_1.png')
    
    rgb = np.concatenate((pre_rgb,cur_rgb),axis=0)
    depth = np.concatenate((pre_depth,cur_depth),axis=0)

    x = np.concatenate((rgb, depth), axis=0)
    x = torch.tensor(x)
    img = x.permute(2,0,1).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=(336, 192))

    # get filename and remove ending
    fname_original = args.image_path.split('/')[-1].split('.')[0]
    args.output_dir = os.path.join(args.output_dir,  f'{args.backbone}_{args.pretrain_backbone}{"_trained" if args.pretrained_weights != "" else ""}', fname_original)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # save original image
    fname = os.path.join(args.output_dir, "img.png")
    img_.save(fname, 'PNG')
    print(f"{fname} saved.")
    # save resized image
    fname = os.path.join(args.output_dir, "img_resized.png")
    print(img.shape)
    plt.imsave(fname, img.squeeze().permute(1,2,0).numpy())
    print(f"{fname} saved.")

    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        print(vars(args), file=f)
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    # img = transform(img)
    # img = transform(img) * 255.
    # ours requires * 255 but not conversion/normalization to RGB 

    # make the image divisible by the patch size
    # w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    # img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save transformed image
    fname = os.path.join(args.output_dir, "transformed_img.png")
    torchvision.utils.save_image(img / 255., fname, normalize=False)
    print(f"{fname} saved.")

    # map to matplotlib colors
    # https://discuss.pytorch.org/t/torch-utils-make-grid-with-cmaps/107471/2
    grid = torchvision.utils.make_grid(torch.tensor(attentions).unsqueeze(1), normalize=True, scale_each=True)
    if args.matplotlib_colors:
        grid = np.apply_along_axis(cm.viridis, 0, grid.numpy()) # converts prediction to cmap!
        grid = torch.from_numpy(np.squeeze(grid))

    # save attentions heatmaps
    fname = os.path.join(args.output_dir, "attn-heads.png")
    torchvision.utils.save_image(grid[0], fname)
    print(f"{fname} saved.")

    for j in range(nh):
        fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    fname = os.path.join(args.output_dir, "attn-heads-sum.png")
    print(np.sum(attentions,axis=0).min(),np.sum(attentions,axis=0).max())
    # exit()
    plt.imsave(fname=fname, arr=np.sum(attentions,axis=0), format='png')
    print(f"{fname} saved.")

    if args.threshold is not None:
        image = torch.tensor(img) # skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)