import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MaskedCrossEntropyLoss(nn.Module):

    def __init__(self, patch_size: int = 16, stride: int = 1, label_smoothing : float = 0.0):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.label_smoothing = label_smoothing

    def forward(self, input, target, mask=None):

        loss = F.cross_entropy(input, target, reduction='none', label_smoothing=self.label_smoothing)

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)
            
            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


class MaskedMSELoss(nn.Module):

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.mse_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean() # Account for zero masks
        else:
            loss = loss.mean() # If this is ever nan, we want it to stop training

        return loss


class MaskedL1Loss(nn.Module):

    def __init__(self, patch_size: int = 16, stride: int = 1, norm_pix=False):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.norm_pix = norm_pix

    def patchify(self, imgs, nh, nw):
        p = self.scale_factor
        x = rearrange(imgs, "b c (nh p1) (nw p2) -> b (nh nw) (p1 p2 c)", nh=nh, nw=nw, p1=p, p2=p)
        return x

    def unpatchify(self, x, nh, nw):
        p = self.scale_factor
        imgs = rearrange(x, "b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)", nh=nh, nw=nw, p1=p, p2=p)
        return imgs

    def forward(self, input, target, mask=None):

        H, W = input.shape[-2:]
        nh, nw = H // self.scale_factor, W // self.scale_factor

        if self.norm_pix:
            target = self.patchify(target, nh, nw)
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            eps = 1e-6
            target = (target - mean) / torch.sqrt(var + eps)
            target = self.unpatchify(target, nh, nw)

        loss = F.l1_loss(input, target, reduction='none')

        if mask is not None:
            if mask.sum() == 0:
                return torch.tensor(0).to(loss.device)

            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)
            loss = loss.mean(dim=1)  # B, C, H, W -> B, H, W
            loss = loss * mask
            # Compute mean per sample
            loss = loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
            loss = loss.nanmean()  # Account for zero masks
        else:
            loss = loss.mean()  # If this is ever nan, we want it to stop training

        return loss


class MaskedMidasLoss(nn.Module):

    def __init__(self, patch_size: int = 16, stride: int = 1, alpha=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.scale_factor = patch_size // stride
        self.alpha = alpha

    @staticmethod
    def masked_shift_and_scale(depth_preds, depth_gt, mask_valid):
        depth_preds_nan = depth_preds.clone()
        depth_gt_nan = depth_gt.clone()
        depth_preds_nan[~mask_valid] = np.nan
        depth_gt_nan[~mask_valid] = np.nan

        mask_diff = mask_valid.view(mask_valid.size()[:2] + (-1,)).sum(-1, keepdims=True) + 1

        t_gt = depth_gt_nan.view(depth_gt_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
        t_gt[torch.isnan(t_gt)] = 0
        diff_gt = torch.abs(depth_gt - t_gt)
        diff_gt[~mask_valid] = 0
        s_gt = (diff_gt.view(diff_gt.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
        depth_gt_aligned = (depth_gt - t_gt) / (s_gt + 1e-6)

        t_pred = depth_preds_nan.view(depth_preds_nan.size()[:2] + (-1,)).nanmedian(-1, keepdims=True)[0].unsqueeze(-1)
        t_pred[torch.isnan(t_pred)] = 0
        diff_pred = torch.abs(depth_preds - t_pred)
        diff_pred[~mask_valid] = 0
        s_pred = (diff_pred.view(diff_pred.size()[:2] + (-1,)).sum(-1, keepdims=True) / mask_diff).unsqueeze(-1)
        depth_pred_aligned = (depth_preds - t_pred) / (s_pred + 1e-6)

        return depth_pred_aligned, depth_gt_aligned

    @staticmethod
    def compute_scale_and_shift(prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / (det[valid] + 1e-6)
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / (det[valid] + 1e-6)

        return x_0, x_1

    @staticmethod
    def reduction_image_based(image_loss, M):
        # mean of average of valid pixels of an image

        # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
        valid = M.nonzero()

        image_loss[valid] = image_loss[valid] / M[valid]

        return torch.mean(image_loss)

    @staticmethod
    def gradient_loss(prediction, target, mask, reduction=reduction_image_based):

        M = torch.sum(mask, (1, 2))

        diff = prediction - target
        diff = torch.mul(mask, diff)

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
        grad_x = torch.mul(mask_x, grad_x)

        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
        mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
        grad_y = torch.mul(mask_y, grad_y)

        image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

        return reduction(image_loss, M)

    @staticmethod
    def gradient_matching_term(prediction, target, mask, scales=4):
            total = 0

            for scale in range(scales):
                step = pow(2, scale)

                total += MaskedMidasLoss.gradient_loss(
                    prediction[:, ::step, ::step],
                    target[:, ::step, ::step],
                    mask[:, ::step, ::step],
                    reduction=MaskedMidasLoss.reduction_image_based
                )

            return total

    def forward(self, input, target, mask=None):

        # Mask
        if mask is None:
            mask = torch.ones_like(input).bool()
        else:
            H, W = input.shape[-2:]
            nh, nw = H // self.scale_factor, W // self.scale_factor
            # Resize mask and upsample
            mask = rearrange(mask, "b (nh nw) -> b nh nw", nh=nh, nw=nw)
            # Convert mask to B, 1, H, W
            mask = F.interpolate(mask.unsqueeze(1).float(), size=(H, W), mode='nearest').bool()

        # Compute SSI Loss
        depth_pred_aligned, depth_gt_aligned = self.masked_shift_and_scale(input, target, mask)
        # TODO: use MSE or L1?
        ssi_loss = F.l1_loss(depth_pred_aligned, depth_gt_aligned, reduction='none')
        ssi_loss = ssi_loss * mask.float()
        # Compute mean per sample
        ssi_loss = ssi_loss.flatten(start_dim=1).sum(dim=1) / mask.flatten(start_dim=1).sum(dim=1)
        ssi_loss = ssi_loss.mean()

        # Compute gradient loss (reg loss)
        prediction_inverse = 1 / (input.squeeze(1) + 1e-6)
        target_inverse = 1 / (target.squeeze(1) + 1e-6)
        scale, shift = self.compute_scale_and_shift(prediction_inverse, target_inverse, mask.squeeze(1))
        prediction_ssi = scale.view(-1, 1, 1) * prediction_inverse + shift.view(-1, 1, 1)
        reg_loss = self.gradient_matching_term(prediction_ssi, target_inverse, mask.squeeze(1))

        loss = ssi_loss + self.alpha * reg_loss


        return loss
