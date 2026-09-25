"""Weak semantic supervision losses migrated from Point2Mask-style training.

This file implements two semantic regularizers:
  1. LAB color prior loss
  2. A tree-filter-inspired RGB affinity smoothing loss

The official Point2Mask tree filter depends on custom CUDA extensions and the
full easymd/mmcv stack. For this project we keep the same training intent using
an in-repo differentiable fallback that only relies on PyTorch.
"""

import torch
import torch.nn.functional as F


def build_partial_target_mask(point_maps):
    """Build a partial 2-class semantic target from sparse fg/bg points."""
    fg = point_maps[:, 0:1].float()
    bg = point_maps[:, 1:2].float()
    target = torch.cat([bg, fg], dim=1).clamp(0.0, 1.0)
    return target


def _unfold_wo_center(x, kernel_size, dilation):
    pad = dilation * (kernel_size // 2)
    unfold = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=pad)
    b, c, h, w = x.shape
    unfold = unfold.view(b, c, kernel_size * kernel_size, h, w)
    center = kernel_size * kernel_size // 2
    return torch.cat([unfold[:, :, :center], unfold[:, :, center + 1:]], dim=2)


def _rgb_to_lab(images):
    """Convert normalized RGB image tensor in [0,1] to LAB."""
    rgb = images.clamp(0.0, 1.0)

    mask = rgb > 0.04045
    rgb = torch.where(mask, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)

    r = rgb[:, 0:1]
    g = rgb[:, 1:2]
    b = rgb[:, 2:3]

    x = 0.412453 * r + 0.357580 * g + 0.180423 * b
    y = 0.212671 * r + 0.715160 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b

    x = x / 0.950456
    z = z / 1.088754

    delta = 6.0 / 29.0

    def f(t):
        return torch.where(t > delta ** 3, t.pow(1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

    fx = f(x)
    fy = f(y)
    fz = f(z)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([l, a, b], dim=1)


def color_prior_loss(logits, images, masks=None, dilation=2, avg_factor=None):
    """Point2Mask-style local color consistency loss in LAB space."""
    if logits.ndim != 4:
        raise ValueError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")

    log_prob = F.log_softmax(logits, dim=1)
    b, _, h, w = logits.shape
    if images.shape[-2:] != (h, w):
        raise ValueError(f"Image/logit size mismatch: {tuple(images.shape)} vs {tuple(logits.shape)}")

    lab = _rgb_to_lab(images)
    kernel_size_list = [3, 5]
    weights = [0.35, 0.65]
    losses = []

    for kernel_size, weight in zip(kernel_size_list, weights):
        log_prob_unfold = _unfold_wo_center(log_prob, kernel_size, dilation)
        log_same_prob = log_prob[:, :, None] + log_prob_unfold
        max_ = log_same_prob.max(dim=1, keepdim=True)[0]
        log_same_prob = (log_same_prob - max_).exp().sum(dim=1).log() + max_.squeeze(1)

        lab_unfold = _unfold_wo_center(lab, kernel_size, dilation)
        lab_diff = lab[:, :, None] - lab_unfold
        lab_sim = (-torch.norm(lab_diff, dim=1) * 0.5).exp()

        loss_weight = (lab_sim >= 0.3).float()
        if masks is not None:
            loss_weight = loss_weight * masks[:, None].float()

        denom = loss_weight.sum(dim=(1, 2, 3)).clamp(min=1.0)
        loss_color = -(log_same_prob * loss_weight).sum(dim=(1, 2, 3)) / denom
        loss_color = loss_color.sum() / (len(loss_color) if avg_factor is None else avg_factor)
        losses.append(weight * loss_color)

    return sum(losses)


def tree_filter_semantic_loss(logits, mask_targets, images, kernels=(7, 11), sigmas=(0.15, 0.30)):
    """Long-range RGB affinity smoothing loss.

    This is a PyTorch fallback for Point2Mask's tree-filter regularizer when the
    official custom op stack is unavailable.
    """
    prob = torch.softmax(logits, dim=1)
    labeled_region = mask_targets.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
    unlabeled_region = 1.0 - labeled_region
    norm = unlabeled_region.sum().clamp(min=1.0)

    total = 0.0
    weight_sum = 0.0

    for kernel_size, sigma in zip(kernels, sigmas):
        prob_unfold = _unfold_wo_center(prob, kernel_size, dilation=1)
        img_unfold = _unfold_wo_center(images, kernel_size, dilation=1)
        img_diff = images[:, :, None] - img_unfold
        img_dist = torch.norm(img_diff, dim=1)
        affinity = torch.exp(-(img_dist ** 2) / max(2.0 * sigma * sigma, 1e-6))
        affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp(min=1e-6)

        smooth_prob = (prob_unfold * affinity[:, None]).sum(dim=2)
        total = total + (torch.abs(prob - smooth_prob) * unlabeled_region).sum()
        weight_sum += 1.0

    return total / (norm * max(weight_sum, 1.0))
