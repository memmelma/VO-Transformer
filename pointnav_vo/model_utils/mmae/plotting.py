import os
import subprocess
import argparse
from tqdm import tqdm
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat

import colorsys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
# from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.data import MetadataCatalog

from mmae import *
from mmae.mmae import pretrain_mmae_base, pretrain_mmae_large
from utils.data_constants import *
from torchvision import datasets, transforms
from utils.dataset_folder import ImageFolder, MultiTaskImageFolder



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

# Cherry-picked ImageNet validation set images
CHERRY_PICKED = [
    34945,25284,38963,10609,14720,17507,45223,44809,43256,21126,
    28791,49127,44061,32268,17741,39783,29284,46568,25980,48345,
    46079,41958,11890,45713,30498,4845,40440,10672,23366,36265,
    17007,787,12905,21497,5289,4825,6432,30805,41163,48324,10900,
    17498,19399,7091, 1956
]

    
class DataAugmentationForMMAE(object):
    def __init__(self, input_size=224, hflip=0.5, imagenet_default_mean_and_std=True):
        self.rgb_mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        self.rgb_std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        self.input_size = input_size
        self.hflip = hflip

    def __call__(self, task_dict):
        flip = random.random() < self.hflip # Stores whether to flip all images or not
        ijhw = None # Stores crop coordinates used for all tasks
        
        # Crop and flip all tasks randomly, but consistently for all tasks
        for task in task_dict:
            if task not in IMAGE_TASKS:
                continue
            if ijhw is None:
                ijhw = transforms.RandomResizedCrop.get_params(
                    task_dict[task], scale=(0.8, 1.0), ratio=(1.0, 1.0)
                )
            i, j, h, w = ijhw
            task_dict[task] = TF.crop(task_dict[task], i, j, h, w)
            task_dict[task] = task_dict[task].resize((self.input_size, self.input_size))
            if flip:
                task_dict[task] = TF.hflip(task_dict[task])
                
        # Convert to Tensor
        for task in task_dict:
            if task in ['depth']:
                img = torch.Tensor(np.array(task_dict[task]) / 2** 16)
                img = img.unsqueeze(0) # 1 x H x W
            elif task in ['rgb']:
                img = TF.to_tensor(task_dict[task])
                img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
            elif task in ['semseg', 'semseg_coco']:
                # TODO: add this to a config instead
                # Rescale to 0.25x size (stride 4)
                scale_factor = 0.25
                img = task_dict[task].resize((int(self.input_size * scale_factor), int(self.input_size * scale_factor)))
                # Using pil_to_tensor keeps it in uint8, to_tensor converts it to float (rescaled to [0, 1])
                img = TF.pil_to_tensor(img).to(torch.long).squeeze(0)
                
            task_dict[task] = img
        
        return task_dict
    
def build_mmae_pretraining_dataset(data_path, domains, input_size, hflip=0.0, imagenet_default_mean_and_std=True):
    transform = DataAugmentationForMMAE(input_size=input_size, hflip=hflip, imagenet_default_mean_and_std=imagenet_default_mean_and_std)
    return MultiTaskImageFolder(data_path, domains, transform=transform)

def denormalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean= [-m/s for m, s in zip(mean, std)],
        std= [1/s for s in std]
    )

def normalize(img, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    return TF.normalize(
        img.clone(),
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD
    )

def get_masked_image(img, mask, image_size=224, patch_size=16, mask_value=0.0):
    img_token = rearrange(
        img.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    img_token[mask.detach().cpu()!=0] = mask_value
    img = rearrange(
        img_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img

def get_pred_with_input(gt, pred, mask, image_size=224, patch_size=16):
    gt_token = rearrange(
        gt.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token = rearrange(
        pred.detach().cpu(), 
        'b c (nh ph) (nw pw) -> b (nh nw) (c ph pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    pred_token[mask.detach().cpu()==0] = gt_token[mask.detach().cpu()==0]
    img = rearrange(
        pred_token, 
        'b (nh nw) (c ph pw) -> b c (nh ph) (nw pw)', 
        ph=patch_size, pw=patch_size, nh=image_size//patch_size, nw=image_size//patch_size
    )
    return img

def plot_semseg_gt(input_dict, ax=None):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(input_dict['rgb'].detach().cpu())[0].permute(1,2,0)
    semseg = F.interpolate(
        input_dict['semseg'].unsqueeze(0).cpu().float(), size=224, mode='nearest'
    ).long()[0,0]
    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()
    
def plot_semseg_gt_masked(input_dict, mask, ax=None, mask_value=1.0):
    img = plot_semseg_gt(input_dict)
    img = torch.LongTensor(img).permute(2,0,1).unsqueeze(0)
    masked_img = get_masked_image(img.float()/255.0, mask, image_size=224, patch_size=16, mask_value=mask_value)
    masked_img = masked_img[0].permute(1,2,0)
    
    if ax is not None:
        ax.imshow(masked_img)
    else:
        return masked_img
    
def plot_semseg_pred(rgb, semseg_preds, ax=None):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(rgb.detach().cpu())[0].permute(1,2,0)
    semseg = F.interpolate(semseg_preds, size=224, mode='nearest')[0].argmax(0).detach().cpu()
    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()
    
def plot_semseg_pred_masked(rgb, semseg_preds, semseg_gt, mask, ax=None):
    metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    instance_mode = ColorMode.IMAGE
    img_viz = 255 * denormalize(rgb.detach().cpu())[0].permute(1,2,0)
    
    semseg = get_pred_with_input(
        semseg_gt.unsqueeze(1), 
        semseg_preds.argmax(1).unsqueeze(1), 
        mask, 
        image_size=56, 
        patch_size=4
    )
    
    semseg = F.interpolate(semseg.float(), size=224, mode='nearest')[0,0].long()
    
    visualizer = Visualizer(img_viz, metadata, instance_mode=instance_mode, scale=1)
    visualizer.draw_sem_seg(semseg)
    if ax is not None:
        ax.imshow(visualizer.get_output().get_image())
    else:
        return visualizer.get_output().get_image()
    
def ax_border(ax, color='black', lw='7'):
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('5')  

def get_random_input_dict(dataset, rand_idx=None, show=False, device='cuda'):
    if rand_idx is None:
        rand_idx = np.random.randint(len(dataset)) 

    input_dict, cl = dataset[rand_idx]
    input_dict = {task: tensor.unsqueeze(0).to(device) for task, tensor in input_dict.items()}

    input_dict['semseg'] = input_dict['semseg_coco']
    del input_dict['semseg_coco']

    # Truncated depth standardization
    if 'depth' in input_dict:
        # Flatten depth and remove bottom and top 10% of values
        trunc_depth = torch.sort(rearrange(input_dict['depth'], 'b c h w -> b (c h w)'), dim=1)[0]
        trunc_depth = trunc_depth[:,int(0.1 * trunc_depth.shape[1]): int(0.9 * trunc_depth.shape[1])]
        input_dict['depth'] = (input_dict['depth'] - trunc_depth.mean(dim=1)[:,None,None,None]) / torch.sqrt(trunc_depth.var(dim=1)[:,None,None,None] + 1e-6)

    if show:
        plt.figure()
        plt.imshow(denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())
        plt.show()
    
    return input_dict, rand_idx


def plot_masked(
        mmae, 
        input_dict, 
        alphas=1.0, 
        num_encoded_tokens=98, 
        sample_tasks_uniformly=False, 
        task_masks=None,
        save_path=None,
        figscale=10.0):

    with torch.no_grad():
        if task_masks is None:
            preds, masks = mmae.forward(
                input_dict, 
                mask_inputs=True, 
                num_encoded_tokens=num_encoded_tokens,
                alphas=alphas,
                sample_tasks_uniformly=sample_tasks_uniformly
            )
        else:
            preds, masks = mmae.forward(
                input_dict, 
                mask_inputs=True, 
                task_masks=task_masks
            )
        preds = {domain: pred.detach().cpu() for domain, pred in preds.items()}
        masks = {domain: mask.detach().cpu() for domain, mask in masks.items()}

    masked_rgb = get_masked_image(
        denormalize(input_dict['rgb']), 
        masks['rgb'],
        image_size=224,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()
    masked_depth = get_masked_image(
        input_dict['depth'], 
        masks['depth'],
        image_size=224,
        mask_value=np.nan
    )[0,0].detach().cpu()

    pred_rgb = denormalize(preds['rgb'])[0].permute(1,2,0).detach().cpu().clamp(0,1)
    pred_depth = preds['depth'][0,0].detach().cpu()

    pred_rgb2 = get_pred_with_input(
        denormalize(input_dict['rgb']), 
        denormalize(preds['rgb']).clamp(0,1), 
        masks['rgb'],
        image_size=224
    )[0].permute(1,2,0).detach().cpu()
    pred_depth2 = get_pred_with_input(
        input_dict['depth'], 
        preds['depth'], 
        masks['depth'],
        image_size=224
    )[0].permute(1,2,0).detach().cpu()

    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.0)

    grid[0].imshow(masked_rgb)
    grid[1].imshow(pred_rgb2)
    grid[2].imshow(denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())

    grid[3].imshow(masked_depth)
    grid[4].imshow(pred_depth2)
    grid[5].imshow(input_dict['depth'][0,0].detach().cpu())

    plot_semseg_gt_masked(input_dict, masks['semseg'], grid[6], mask_value=1.0)
    plot_semseg_pred_masked(input_dict['rgb'], preds['semseg'], input_dict['semseg'], masks['semseg'], grid[7])
    plot_semseg_gt(input_dict, grid[8])

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_single_to_two(mmae, input_dict, save_path=None, figscale=10.0):

    with torch.no_grad():
        preds_rgb, _ = mmae.forward({'rgb': input_dict['rgb']}, mask_inputs=False)
        preds_depth, _ = mmae.forward({'depth': input_dict['depth']}, mask_inputs=False)
        preds_semseg, _ = mmae.forward({'semseg': input_dict['semseg']}, mask_inputs=False)
    
    pred_depth2rgb = denormalize(preds_depth['rgb'])[0].permute(1,2,0).detach().cpu().clamp(0,1)
    pred_semseg2rgb = denormalize(preds_semseg['rgb'])[0].permute(1,2,0).detach().cpu().clamp(0,1)

    pred_rgb2depth = preds_rgb['depth'][0,0].detach().cpu()
    pred_depth2depth = preds_depth['depth'][0,0].detach().cpu()
    pred_semseg2depth = preds_semseg['depth'][0,0].detach().cpu()
    
    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.0)

    # RGB->X
    grid[0].imshow(denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())
    grid[1].imshow(pred_rgb2depth)
    plot_semseg_pred(input_dict['rgb'], preds_rgb['semseg'], grid[2])

    # Depth->X
    grid[3].imshow(input_dict['depth'][0,0].detach().cpu())
    grid[4].imshow(pred_depth2rgb)
    plot_semseg_pred(input_dict['rgb'], preds_depth['semseg'], grid[5])
    
    # Semseg->X
    plot_semseg_gt(input_dict, grid[6])
    grid[7].imshow(pred_semseg2rgb)
    grid[8].imshow(pred_semseg2depth)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_two_to_one(mmae, input_dict, save_path=None, figscale=10.0):

    with torch.no_grad():
        preds_rgb_depth, _ = mmae.forward({'rgb': input_dict['rgb'], 'depth': input_dict['depth']}, mask_inputs=False)
        preds_rgb_semseg, _ = mmae.forward({'rgb': input_dict['rgb'], 'semseg': input_dict['semseg']}, mask_inputs=False)
        preds_depth_semseg, _ = mmae.forward({'depth': input_dict['depth'], 'semseg': input_dict['semseg']}, mask_inputs=False)
    
    pred_depth_semseg2rgb = denormalize(preds_depth_semseg['rgb'])[0].permute(1,2,0).detach().cpu().clamp(0,1)
    pred_rgb_semseg2depth = preds_rgb_semseg['depth'][0,0].detach().cpu()

    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0.0)

    # Inputs
    grid[0].imshow(denormalize(input_dict['rgb'])[0].permute(1,2,0).detach().cpu())
    grid[1].imshow(input_dict['depth'][0,0].detach().cpu())
    plot_semseg_gt(input_dict, grid[2])

    # Preds
    grid[3].imshow(pred_depth_semseg2rgb)
    grid[4].imshow(pred_rgb_semseg2depth)
    plot_semseg_pred(input_dict['rgb'], preds_rgb_depth['semseg'], grid[5])
    
    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_single_patch_pred(
        mmae, 
        input_dict, 
        x, y, 
        selected_task, 
        patch_size = 16,
        save_path=None, 
        figscale=7.0,
        fontsize=12):
    
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size
    
    xy_idxs = {'rgb': [], 'depth': [], 'semseg': []}
    xy_idxs[selected_task].append([y,x])

    task_masks = mmae.make_mask(
        N_H, N_W, xy_idxs, indicate_visible=True, 
        full_tasks=[task for task in input_dict.keys() if task != selected_task],
        device='cuda'
    )

    with torch.no_grad():
        preds, _ = mmae.forward(input_dict, task_masks=task_masks)


    masked_rgb = get_masked_image(
        denormalize(input_dict['rgb']), 
        task_masks['rgb'],
        image_size=224,
        mask_value=1.0
    )[0].permute(1,2,0).detach().cpu()
    masked_depth = get_masked_image(
        input_dict['depth'], 
        task_masks['depth'],
        image_size=224,
        mask_value=np.nan
    )[0,0].detach().cpu()


    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.0)

    grid[0].imshow(masked_rgb)
    grid[0].set_title('RGB input', fontsize=fontsize)

    grid[1].imshow(masked_depth)
    grid[1].set_title('Depth input', fontsize=fontsize)

    plot_semseg_gt_masked(input_dict, task_masks['semseg'], grid[2], mask_value=1.0)
    grid[2].set_title('Semantic input', fontsize=fontsize)

    if selected_task == 'rgb':
        pred_rgb = get_pred_with_input(
            denormalize(input_dict['rgb']), 
            denormalize(preds['rgb']).clamp(0,1), 
            task_masks['rgb'],
            image_size=224
        )[0].permute(1,2,0).detach().cpu()
        grid[3].imshow(pred_rgb)
        grid[3].set_title('RGB prediction', fontsize=fontsize)

    elif selected_task == 'depth':
        pred_depth = get_pred_with_input(
            input_dict['depth'], 
            preds['depth'], 
            task_masks['depth'],
            image_size=224
        )[0].permute(1,2,0).detach().cpu()
        grid[3].imshow(pred_depth)
        grid[3].set_title('Depth prediction', fontsize=fontsize)

    elif selected_task == 'semseg':
        plot_semseg_pred_masked(
            input_dict['rgb'], preds['semseg'], 
            input_dict['semseg'], task_masks['semseg'], grid[3]
        )
        grid[3].set_title('Semantic prediction', fontsize=fontsize)

    rect = patches.Rectangle(
        (x*patch_size, y*patch_size), patch_size, patch_size, 
        linewidth=1, edgecolor='black', facecolor='none'
    )
    grid[3].add_patch(rect)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])

    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def make_video(frames_pattern, video_path, fps=10, bitrate='500k'):
    cmd = f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{frames_pattern}' \
           -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white' \
           -c:v libx264 -b:v {bitrate} -pix_fmt yuv420p {video_path}"
    
    subprocess.call(cmd, shell=True)


def animate_single_patch_pred(mmae, input_dict, save_dir, patch_size=16, show_progress=True, fps=10):
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size
    
    for selected_task in input_dict.keys():
        if show_progress:
            pbar = tqdm(total=N_H * N_W, desc=f'Plotting {selected_task}')
        for y in range(N_H):
            for x in range(N_W):
                anim_idx = N_H * y + x
                save_path = os.path.join(save_dir, 'frames', selected_task, f'{anim_idx:05d}.jpg')
                plot_single_patch_pred(mmae, input_dict, x, y, selected_task, save_path=save_path, patch_size=patch_size)
                if show_progress:
                    pbar.update(1)
        if show_progress:
            pbar.close()
            
        make_video(
            os.path.join(save_dir, 'frames', selected_task, '*.jpg'), 
            os.path.join(save_dir, f'{selected_task}.mp4'), 
            fps=fps
        )
        
    subprocess.call(f"rm -rf {os.path.join(save_dir, 'frames')}", shell=True)


def plot_randomly_perturbed_semseg(
        mmae, 
        input_dict,
        predict_task='rgb',
        only_semseg_input=False,
        save_path=None, 
        patch_size=16,
        figscale=10.0,
        fontsize=15,
        seed=0):
    
    in_dict = {k: v.clone() for k,v in input_dict.items()}
    
    np.random.seed(seed)
    unique_classes = torch.unique(in_dict['semseg'])
    rand_class = unique_classes[np.random.randint(len(unique_classes))].item()
    in_dict['semseg'][in_dict['semseg'] == rand_class] = np.random.randint(133)
    
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size

    xy_idxs = {'rgb': [], 'depth': [], 'semseg': []}
    
    if only_semseg_input:
        full_tasks = ['semseg']
    else:
        full_tasks = ['semseg', 'depth'] if predict_task == 'rgb' else ['semseg', 'rgb']
    task_masks = mmae.make_mask(
        N_H, N_W, xy_idxs, indicate_visible=True, 
        full_tasks=full_tasks,
        device='cuda'
    )

    with torch.no_grad():
        preds, _ = mmae.forward(in_dict, task_masks=task_masks)
        
    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, len(full_tasks)+2), axes_pad=0.0)

    grid_offset = -1 if only_semseg_input else 0

    if predict_task == 'rgb':
        if not only_semseg_input:
            grid[0].imshow(in_dict['depth'][0,0].detach().cpu())
            grid[0].set_title('Depth input', fontsize=fontsize)
        
        grid[2+grid_offset].imshow(denormalize(preds['rgb'])[0].permute(1,2,0).detach().cpu())
        grid[2+grid_offset].set_title('RGB pred', fontsize=fontsize)

        grid[3+grid_offset].imshow(denormalize(in_dict['rgb'])[0].permute(1,2,0).detach().cpu())
        grid[3+grid_offset].set_title('RGB original', fontsize=fontsize)
        
    else:
        if not only_semseg_input:
            grid[0].imshow(denormalize(in_dict['rgb'])[0].permute(1,2,0).detach().cpu())
            grid[0].set_title('RGB input', fontsize=fontsize)
        
        grid[2+grid_offset].imshow(preds['depth'][0,0].detach().cpu())
        grid[2+grid_offset].set_title('Depth pred', fontsize=fontsize)

        grid[3+grid_offset].imshow(in_dict['depth'][0,0].detach().cpu())
        grid[3+grid_offset].set_title('Depth original', fontsize=fontsize)

    plot_semseg_gt(in_dict, grid[1+grid_offset])
    grid[1+grid_offset].set_title('Semantic input', fontsize=fontsize)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_scaled_depth(
        mmae, 
        input_dict,
        raise_to_pow=1,
        exponentiate=False,
        save_path=None, 
        patch_size=16,
        figscale=10.0,
        fontsize=9):
    
    in_dict = {k: v.clone() for k,v in input_dict.items()}
    
    scale_string = 'd'
    if raise_to_pow > 1:
        in_dict['depth'] = torch.pow(in_dict['depth'], raise_to_pow)
        scale_string += f'^{raise_to_pow}'
    if exponentiate:
        in_dict['depth'] = torch.exp(in_dict['depth'])
        scale_string = 'e^{' + scale_string + '}'
    in_dict['depth'] = (in_dict['depth'] - in_dict['depth'].mean()) / in_dict['depth'].std()
    
    
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size

    xy_idxs = {'rgb': [], 'depth': [], 'semseg': []}    
    task_masks = mmae.make_mask(
        N_H, N_W, xy_idxs, indicate_visible=True, 
        full_tasks=['depth'],
        device='cuda'
    )

    with torch.no_grad():
        preds, _ = mmae.forward(in_dict, task_masks=task_masks)
        
    fig = plt.figure(figsize=(figscale, figscale))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 5), axes_pad=0.0)
    
    grid[0].imshow(in_dict['depth'][0,0].detach().cpu())
    grid[0].set_title(rf'Scaled depth input: std(${scale_string}$)', fontsize=fontsize)

    grid[1].imshow(denormalize(preds['rgb'])[0].permute(1,2,0).detach().cpu())
    grid[1].set_title('RGB prediction', fontsize=fontsize)

    grid[2].imshow(denormalize(in_dict['rgb'])[0].permute(1,2,0).detach().cpu())
    grid[2].set_title('RGB original', fontsize=fontsize)

    plot_semseg_pred(input_dict['rgb'], preds['semseg'], grid[3])
    grid[3].set_title('Semantic prediction', fontsize=fontsize)

    plot_semseg_gt(in_dict, grid[4])
    grid[4].set_title('Semantic original', fontsize=fontsize)

    for ax in grid:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def generate_progressive_masks(N_H, N_W, device='cuda'):
    xy_idxs = {'rgb': [], 'depth': [], 'semseg': []}    
    task_masks = mmae.make_mask(
        N_H, N_W, xy_idxs, indicate_visible=True, 
        full_tasks=[],
        device=device
    )

    progressive_masks = [task_masks]

    for i in range(sum([v.shape[1] for v in task_masks.values()])):
        next_mask = {k: v.clone() for k,v in progressive_masks[-1].items()}

        # Select which task to add a patch to
        valid_tasks = [k for k,v in next_mask.items() if (v==0).sum() < v.shape[1]]
        next_task = valid_tasks[np.random.randint(len(valid_tasks))]

        valid_idxs = torch.where(next_mask[next_task] != 0)[1]
        next_token = valid_idxs[np.random.randint(len(valid_idxs))]

        next_mask[next_task][:,next_token] = 0

        progressive_masks.append(next_mask)
        
    return progressive_masks


def animate_progressive_masking(mmae, input_dict, save_dir, patch_size=16, show_progress=True, fps=15):
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size
    
    progressive_masks = generate_progressive_masks(N_H, N_W)
    
    if show_progress:
        pbar = tqdm(total=len(progressive_masks), desc=f'Plotting progressive masking')
    for anim_idx, task_mask in enumerate(progressive_masks):
        save_path = os.path.join(save_dir, 'frames', f'{anim_idx:05d}.jpg')
        plot_masked(mmae, input_dict, task_masks=task_mask, save_path=save_path)
        
        if show_progress:
            pbar.update(1)
    if show_progress:
        pbar.close()
    
    make_video(
        os.path.join(save_dir, 'frames', '*.jpg'), 
        os.path.join(save_dir, f'progressive_masking.mp4'), 
        fps=fps,
        bitrate='2500k'
    )


def generate_mask_transitions(N_H, N_W, num_visible=98, same_task_masks=False, device='cuda'):
    '''
    Move num_visible tokens from RGB->Depth->Semseg->RGB
    '''
    
    rgb_idxs = np.random.choice(N_H * N_W, size=num_visible, replace=False)
    if same_task_masks:
        depth_idxs = rgb_idxs
        semseg_idxs = rgb_idxs
    else:
        depth_idxs = np.random.choice(N_H * N_W, size=num_visible, replace=False)
        semseg_idxs = np.random.choice(N_H * N_W, size=num_visible, replace=False)

    move_order_from = np.random.permutation(num_visible)
    if same_task_masks:
        move_order_to = move_order_from
    else:
        move_order_to = np.random.permutation(num_visible)

    initial_mask = {k: torch.ones(1, N_H * N_W).long().to(device) for k in ['rgb', 'depth', 'semseg']}
    initial_mask['rgb'][:,rgb_idxs] = 0
    progressive_masks = [initial_mask]

    # Move RGB -> Depth
    for idx in range(num_visible):
        from_idx = rgb_idxs[move_order_from[idx]]
        to_idx = depth_idxs[move_order_to[idx]]
        new_mask = {k: v.clone() for k, v in progressive_masks[-1].items()}
        new_mask['rgb'][:,from_idx] = 1
        new_mask['depth'][:,to_idx] = 0
        progressive_masks.append(new_mask)

    # Move Depth -> Semseg
    for idx in range(num_visible):
        from_idx = depth_idxs[move_order_from[idx]]
        to_idx = semseg_idxs[move_order_to[idx]]
        new_mask = {k: v.clone() for k, v in progressive_masks[-1].items()}
        new_mask['depth'][:,from_idx] = 1
        new_mask['semseg'][:,to_idx] = 0
        progressive_masks.append(new_mask)

    # Move Semseg -> RGB
    for idx in range(num_visible):
        from_idx = semseg_idxs[move_order_from[idx]]
        to_idx = rgb_idxs[move_order_to[idx]]
        new_mask = {k: v.clone() for k, v in progressive_masks[-1].items()}
        new_mask['semseg'][:,from_idx] = 1
        new_mask['rgb'][:,to_idx] = 0
        progressive_masks.append(new_mask)
        
    return progressive_masks[:-1]

def animate_mask_transitions(mmae, input_dict, save_dir, num_visible=98, same_task_masks=False, patch_size=16, show_progress=True, fps=15):
    N_H = input_dict['rgb'].shape[-2] // patch_size
    N_W = input_dict['rgb'].shape[-1] // patch_size
    
    mask_transitions = generate_mask_transitions(N_H, N_W, num_visible=num_visible, same_task_masks=same_task_masks, device='cuda')
    
    if show_progress:
        pbar = tqdm(total=len(mask_transitions), desc=f'Plotting mask transitions')
    for anim_idx, task_mask in enumerate(mask_transitions):
        save_path = os.path.join(save_dir, 'frames', f'{anim_idx:05d}.jpg')
        plot_masked(mmae, input_dict, task_masks=task_mask, save_path=save_path)
        
        if show_progress:
            pbar.update(1)
    if show_progress:
        pbar.close()
    
    file_name = 'mask_transition_same' if same_task_masks else 'mask_transition_rand'
    file_name += f'_{num_visible}'
    make_video(
        os.path.join(save_dir, 'frames', '*.jpg'), 
        os.path.join(save_dir, f'{file_name}.mp4'), 
        fps=fps,
        bitrate='2500k'
    )
    
    subprocess.call(f"rm -rf {os.path.join(save_dir, 'frames')}", shell=True)



rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)

def shift_hue(arr, hout):
    r, g, b, a = np.rollaxis(arr, axis=-1)
    h, s, v = rgb_to_hsv(r, g, b)
    h = hout
    r, g, b = hsv_to_rgb(h, s, v)
    arr = np.dstack((r, g, b, a))
    return arr

def colorize(image, hue):
    """
    Colorize PIL image `original` with the given
    `hue` (hue within 0-360); returns another PIL image.
    """
    img = image.convert('RGBA')
    arr = np.array(np.asarray(img).astype('float'))
    new_img = Image.fromarray(shift_hue(arr, hue/360.).astype('uint8'), 'RGBA')

    return new_img

def get_hue_shifted_input_dict(input_dict, hue_shift=0.0, xy=None, patch_size=16):
    '''

    :param xy: Specify xy location of single patch to change. If None, entire image is shifted.
    '''
    img_pil = Image.fromarray((
        denormalize(
            input_dict['rgb'][0].detach().cpu().clone()
        ).permute(1,2,0).numpy() * 255
    ).astype(np.uint8))
    shifted = colorize(img_pil, hue_shift + 90) # hue shift 90 results in no change
    shifted = normalize(torch.Tensor(np.array(shifted))[:,:,:3].permute(2,0,1).unsqueeze(0) / 255)
    shifted = shifted.to(input_dict['rgb'].device)

    shifted_dict = {k: v.clone() for k,v in input_dict.items()}
    if xy is None:
        shifted_dict['rgb'] = shifted
    else:
        x_from = xy[0] * patch_size
        x_to = x_from + patch_size
        y_from = xy[1] * patch_size
        y_to = y_from + patch_size
        shifted_dict['rgb'][:,:,y_from:y_to,x_from:x_to] = shifted[:,:,y_from:y_to,x_from:x_to]
    return shifted_dict





def get_args():
    parser = argparse.ArgumentParser(description='Plotting Config', add_help=False)
    
    parser.add_argument('--data_root', default='/datasets/imagenet_multitask/val/', type=str,
                        help='Root directory of multi-task dataset')
    parser.add_argument('--weights', 
                        default='/datasets/home/roman/MMAE_seg/output/pretrain/B_98_rgb+-depth-semseg_E1600/checkpoint-1599.pth', 
                        type=str, help='Path to MultiMAE model weights')
    parser.add_argument('--save_dir', default='./plots/', type=str,
                        help='Root directory for saving plots and videos')
    
    parser.add_argument('--num_rand_imgs', default=100, type=int,
                        help='Number of random images to plot')
    parser.add_argument('--num_cherry_imgs', default=0, type=int,
                        help='Number of cherry-picked images to plot')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--num_random_variants', default=10, type=int,
                        help='Number of random variants to plot')
    

    parser.add_argument('--domains', default='rgb-depth-semseg', type=str,
                        help='Task/modality names, separated by hyphen')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='base patch size for image-like modalities')
    parser.add_argument('--model_size', default='base', type=str,
                        help='ViT backbone size. base or large.')
    parser.add_argument('--decoder_dim', default=256, type=int,
                        help='Token dimension inside the decoder layers')
    parser.add_argument('--decoder_depth', default=2, type=int,
                        help='Number of self-attention layers after the initial cross attention')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    args.domains = args.domains.split('-')

    # Setup dataset
    data_loader_domains = [task for task in args.domains if task != 'semseg']
    if 'semseg' in args.domains:
        data_loader_domains.append('semseg_coco')
    dataset = build_mmae_pretraining_dataset(
        args.data_root, 
        data_loader_domains,
        input_size=args.input_size, 
        hflip=0.5
    )


    # Setup model
    input_adapters = {
        domain: dinfo['input_adapter'](
            patch_size_full = args.patch_size,
        )
        for domain, dinfo in DOMAIN_CONF.items()
        if domain in args.domains
    }
    output_adapters = {
        domain: dinfo['output_adapter'](
            patch_size_full = args.patch_size,
            dim_tokens = args.decoder_dim,
            use_task_queries = True,
            depth = args.decoder_depth,
            context_tasks=args.domains,
            task=domain
        )
        for domain, dinfo in DOMAIN_CONF.items()
        if domain in args.domains
    }
    if args.model_size == 'base':
        mmae = pretrain_mmae_base(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
        )
        size_id = 'B'
    else:
        mmae = pretrain_mmae_large(
            input_adapters=input_adapters,
            output_adapters=output_adapters,
        )
        size_id = 'L'

    # Load MultiMAE checkpoint
    ckpt = torch.load(args.weights, map_location='cpu')
    mmae.load_state_dict(ckpt['model'], strict=False)
    mmae = mmae.eval()
    mmae = mmae.cuda()


    # Based on arguments, sample random or use cherry-picked images
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idxs = CHERRY_PICKED[:args.num_cherry_imgs]
    idxs += np.random.randint(0, len(dataset), args.num_rand_imgs).tolist()

    # Run plotting for selected indices
    for prog_idx, idx in enumerate(idxs):
        print(f'Plotting img {prog_idx+1}/{len(idxs)} (idx: {idx})')

        # Load random image
        input_dict, _ = get_random_input_dict(dataset, rand_idx=idx)

        # # Plot multiple samples of randomized masks
        # print(f'Saving {args.num_random_variants} random mask images')
        # for sample_nb in range(args.num_random_variants):
        #     save_path = os.path.join(args.save_dir, f'{size_id}_randmask_a1.0', f'{idx}_{sample_nb}.pdf')
        #     plot_masked(
        #         mmae, input_dict, 
        #         alphas=1.0, num_encoded_tokens=98, 
        #         sample_tasks_uniformly=False, save_path=save_path
        #     )

        # # Plot full single input to two other tasks preds
        # print('Saving full single-input plot')
        # save_path = os.path.join(args.save_dir, f'{size_id}_one2two', f'{idx}.pdf')
        # plot_single_to_two(mmae, input_dict, save_path=save_path)

        # # Plot two full inputs to remaining task preds
        # print('Saving full dual-input plot')
        # save_path = os.path.join(args.save_dir, f'{size_id}_two2one', f'{idx}.pdf')
        # plot_two_to_one(mmae, input_dict, save_path=save_path)

        # # Create sliding visible patch animation
        # print('Creating sliding visible patch animation')
        # save_path = os.path.join(args.save_dir, f'{size_id}_single_patch_pred', str(idx))
        # animate_single_patch_pred(mmae, input_dict, save_path, patch_size=args.patch_size)

        # Plot random changes to semantic input
        print('Plotting random changes to semantic input')
        for sample_nb in range(args.num_random_variants):
            save_path = os.path.join(args.save_dir, f'{size_id}_rand_semseg_perturb', f'{idx}_rgb_{sample_nb}.pdf')
            plot_randomly_perturbed_semseg(
                mmae, input_dict, predict_task='rgb', save_path=save_path, seed=idx+sample_nb, patch_size=args.patch_size
            )
            save_path = os.path.join(args.save_dir, f'{size_id}_rand_semseg_perturb', f'{idx}_depth_{sample_nb}.pdf')
            plot_randomly_perturbed_semseg(
                mmae, input_dict, predict_task='depth', save_path=save_path, seed=idx+sample_nb, patch_size=args.patch_size
            )
            save_path = os.path.join(args.save_dir, f'{size_id}_rand_semseg_perturb', f'{idx}_semseg2rgb_{sample_nb}.pdf')
            plot_randomly_perturbed_semseg(
                mmae, input_dict, predict_task='rgb', only_semseg_input=True, save_path=save_path, seed=idx+sample_nb, patch_size=args.patch_size
            )
            save_path = os.path.join(args.save_dir, f'{size_id}_rand_semseg_perturb', f'{idx}_semseg2depth_{sample_nb}.pdf')
            plot_randomly_perturbed_semseg(
                mmae, input_dict, predict_task='depth', only_semseg_input=True, save_path=save_path, seed=idx+sample_nb, patch_size=args.patch_size
            )

        # # Plot scaled depth as single input
        # print('Plotting scaled depth as single input')
        # for raise_to_pow in range(1,8):
        #     save_path = os.path.join(args.save_dir, f'{size_id}_depth_perturb', f'{idx}_pow{raise_to_pow}.pdf')
        #     plot_scaled_depth(
        #         mmae, input_dict, raise_to_pow=raise_to_pow, exponentiate=False, save_path=save_path, patch_size=args.patch_size
        #     )        
        # save_path = os.path.join(args.save_dir, f'{size_id}_depth_perturb', f'{idx}_exp.pdf')
        # plot_scaled_depth(
        #     mmae, input_dict, raise_to_pow=1, exponentiate=True, save_path=save_path, patch_size=args.patch_size
        # )  

        # # Plot progressive masking
        # print('Plotting progressive masking')
        # save_path = os.path.join(args.save_dir, f'{size_id}_prog_masking', str(idx))
        # animate_progressive_masking(mmae, input_dict, save_path, show_progress=True, fps=15)

        # # Plot mask transitions
        # print('Plotting mask transitions')
        # save_path = os.path.join(args.save_dir, f'{size_id}_mask_transitions', str(idx))
        # animate_mask_transitions(mmae, input_dict, save_path, num_visible=98, same_task_masks=False, show_progress=True, fps=15)
        # animate_mask_transitions(mmae, input_dict, save_path, num_visible=98, same_task_masks=True, show_progress=True, fps=15)