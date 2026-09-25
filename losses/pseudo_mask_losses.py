"""BCE + Dice loss for soft pseudo-mask supervision."""

import torch
import torch.nn.functional as F


def binary_bce_dice_loss(logits, target, eps=1e-6):
    """BCE + Dice loss with soft target.

    Args:
        logits: [B, 1, H, W]  network mask logits
        target: [B, 1, H, W]  soft pseudo-mask (detached)

    Returns:
        scalar loss
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)

    prob = torch.sigmoid(logits)
    prob_flat = prob.reshape(prob.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    intersection = (prob_flat * target_flat).sum(dim=1)
    dice = 1.0 - (2.0 * intersection + eps) / (
        prob_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)

    return bce + dice.mean()


def generalized_dice_loss(logits, target, eps=1e-6):
    """Generalized Dice Loss for binary segmentation with soft targets.

    Uses inverse-squared class-volume weighting to reduce sensitivity to class
    imbalance. For binary segmentation we form 2-class probabilities:
      bg = 1 - sigmoid(logits), fg = sigmoid(logits).
    """
    prob_fg = torch.sigmoid(logits)
    prob_bg = 1.0 - prob_fg
    prob = torch.cat([prob_bg, prob_fg], dim=1)  # [B, 2, H, W]

    target_fg = target.clamp(0.0, 1.0)
    target_bg = 1.0 - target_fg
    target_oh = torch.cat([target_bg, target_fg], dim=1)  # [B, 2, H, W]

    prob_flat = prob.reshape(prob.shape[0], prob.shape[1], -1)
    target_flat = target_oh.reshape(target_oh.shape[0], target_oh.shape[1], -1)

    class_volume = target_flat.sum(dim=2)
    weights = 1.0 / (class_volume * class_volume + eps)

    intersection = (prob_flat * target_flat).sum(dim=2)
    denom = (prob_flat + target_flat).sum(dim=2)

    numerator = 2.0 * (weights * intersection).sum(dim=1)
    denominator = (weights * denom).sum(dim=1) + eps
    loss = 1.0 - numerator / denominator
    return loss.mean()
