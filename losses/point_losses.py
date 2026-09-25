"""Point-level supervision losses: semantic CE and mask BCE at clicked positions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def point_level_ce_loss(semantic_logits, point_maps, ignore_index=255, eps=1e-6):
    """Cross-entropy at clicked point positions.

    point_maps[:, 0] = foreground points  -> label 1 (polyp)
    point_maps[:, 1] = background points  -> label 0 (background)

    Overlapping fg/bg points cause an assertion error.

    Args:
        semantic_logits: [B, 2, H, W]
        point_maps:       [B, 2, H, W]  ch0=fg, ch1=bg

    Returns:
        scalar loss
    """
    B, _, H, W = point_maps.shape
    fg_mask = point_maps[:, 0] > 0  # [B, H, W]
    bg_mask = point_maps[:, 1] > 0  # [B, H, W]

    overlap = fg_mask & bg_mask
    assert not overlap.any(), "Foreground and background point maps overlap"

    label = torch.full((B, H, W), ignore_index, dtype=torch.long, device=point_maps.device)
    label[fg_mask] = 1
    label[bg_mask] = 0

    valid = (fg_mask | bg_mask).sum()
    if valid == 0:
        return torch.tensor(0.0, device=point_maps.device, requires_grad=True)

    return F.cross_entropy(semantic_logits, label, ignore_index=ignore_index)


def point_mask_loss(mask_logits, point_maps, eps=1e-6):
    """BCE at clicked points: fg points → 1, bg points → 0.

    Args:
        mask_logits: [B, 1, H, W]
        point_maps:  [B, 2, H, W]  ch0=fg, ch1=bg

    Returns:
        scalar loss
    """
    fg_mask = point_maps[:, 0:1] > 0  # [B, 1, H, W]
    bg_mask = point_maps[:, 1:2] > 0  # [B, 1, H, W]
    valid = fg_mask | bg_mask

    if valid.sum() == 0:
        return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

    target = torch.zeros_like(mask_logits)
    target[fg_mask] = 1.0

    loss = F.binary_cross_entropy_with_logits(
        mask_logits[valid], target[valid], reduction='mean'
    )
    return loss


def partial_cross_entropy_loss(mask_logits, point_maps, ignore_index=255):
    """Partial cross-entropy on clicked positions for a binary mask head.

    This converts the 1-channel mask logits into 2-class logits [-z, z] and
    applies CE only at annotated fg/bg points.
    """
    del ignore_index

    fg_mask = point_maps[:, 0] > 0  # [B, H, W]
    bg_mask = point_maps[:, 1] > 0  # [B, H, W]
    valid = fg_mask | bg_mask

    if valid.sum() == 0:
        return torch.tensor(0.0, device=mask_logits.device, requires_grad=True)

    binary_logits = torch.cat([-mask_logits, mask_logits], dim=1)  # [B, 2, H, W]
    label = torch.zeros_like(point_maps[:, 0], dtype=torch.long)
    label[fg_mask] = 1

    logits_flat = binary_logits.permute(0, 2, 3, 1)[valid]
    label_flat = label[valid]
    return F.cross_entropy(logits_flat, label_flat)
