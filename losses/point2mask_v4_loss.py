"""Two-stage closed-loop loss for V4."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_affinity_loss import boundary_affinity_loss
from .point_losses import point_level_ce_loss, point_mask_loss, partial_cross_entropy_loss
from .pseudo_mask_losses import binary_bce_dice_loss, generalized_dice_loss
from .semantic_weak_losses import (
    build_partial_target_mask,
    color_prior_loss,
    tree_filter_semantic_loss,
)


class Point2MaskV4Loss(nn.Module):
    def __init__(
        self,
        lambda_sem=1.0,
        lambda_mask=1.0,
        lambda_boundary=0.2,
        lambda_point_mask=0.5,
        lambda_sem_color=0.05,
        lambda_sem_tree=0.03,
        lambda_aux_mask=0.4,
        use_partial_mask_ce=True,
        use_generalized_dice=True,
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_mask = lambda_mask
        self.lambda_boundary = lambda_boundary
        self.lambda_point_mask = lambda_point_mask
        self.lambda_sem_color = lambda_sem_color
        self.lambda_sem_tree = lambda_sem_tree
        self.lambda_aux_mask = lambda_aux_mask
        self.use_partial_mask_ce = use_partial_mask_ce
        self.use_generalized_dice = use_generalized_dice

    def _semantic_weak_loss(self, semantic_logits, point_maps, image):
        loss_color = semantic_logits.new_tensor(0.0)
        loss_tree = semantic_logits.new_tensor(0.0)
        if image is None or (self.lambda_sem_color <= 0 and self.lambda_sem_tree <= 0):
            return loss_color, loss_tree

        image_small = image
        if image_small.shape[-2:] != semantic_logits.shape[-2:]:
            image_small = F.interpolate(image_small, size=semantic_logits.shape[-2:], mode="bilinear", align_corners=False)
        partial_target = build_partial_target_mask(point_maps)
        if partial_target.shape[-2:] != semantic_logits.shape[-2:]:
            partial_target = F.interpolate(partial_target, size=semantic_logits.shape[-2:], mode="nearest")
        unlabeled_mask = (partial_target.sum(dim=1) == 0).float()

        if self.lambda_sem_color > 0:
            loss_color = color_prior_loss(semantic_logits, image_small, masks=unlabeled_mask)
        if self.lambda_sem_tree > 0:
            loss_tree = tree_filter_semantic_loss(semantic_logits, partial_target, image_small)
        return loss_color, loss_tree

    def forward(self, outputs, point_maps, image=None, gt_mask=None):
        del gt_mask

        P0 = outputs["mask_logits0"]
        P1 = outputs["mask_logits"]
        Ps0 = outputs["semantic_logits0"]
        Ps1 = outputs["semantic_logits"]
        Pb0 = outputs["boundary_logits0"]
        Pb1 = outputs["boundary_logits"]
        T1 = outputs["T1"].detach()
        TR = outputs["TR"].detach()
        T1_target = T1[:, 1:2]
        TR_target = TR[:, 1:2]
        stage2_weight = float(outputs.get("v4_stage2_weight", P1.new_tensor(1.0)).item())
        stage1_weight = 1.0 - stage2_weight
        use_ppot = bool(outputs.get("use_ppot", True))

        if not use_ppot:
            loss_mask_aux = P0.new_tensor(0.0)
            loss_mask_final = P1.new_tensor(0.0)
        elif self.use_generalized_dice:
            loss_mask_aux = generalized_dice_loss(P0, T1_target)
            loss_mask_final = generalized_dice_loss(P1, TR_target)
        else:
            loss_mask_aux = binary_bce_dice_loss(P0, T1_target)
            loss_mask_final = binary_bce_dice_loss(P1, TR_target)

        if self.use_partial_mask_ce:
            loss_point_mask0 = partial_cross_entropy_loss(P0, point_maps)
            loss_point_mask1 = partial_cross_entropy_loss(P1, point_maps)
        else:
            loss_point_mask0 = point_mask_loss(P0, point_maps)
            loss_point_mask1 = point_mask_loss(P1, point_maps)
        loss_sem0 = point_level_ce_loss(Ps0, point_maps)
        loss_sem1 = point_level_ce_loss(Ps1, point_maps)
        if use_ppot:
            loss_boundary0 = boundary_affinity_loss(Pb0, T1)
            loss_boundary1 = boundary_affinity_loss(Pb1, TR)
        else:
            loss_boundary0 = Pb0.new_tensor(0.0)
            loss_boundary1 = Pb1.new_tensor(0.0)

        loss_sem_color0, loss_sem_tree0 = self._semantic_weak_loss(Ps0, point_maps, image)
        loss_sem_color1, loss_sem_tree1 = self._semantic_weak_loss(Ps1, point_maps, image)
        loss_point_mask = stage1_weight * loss_point_mask0 + stage2_weight * loss_point_mask1
        loss_sem = stage1_weight * loss_sem0 + stage2_weight * loss_sem1
        loss_boundary = stage1_weight * loss_boundary0 + stage2_weight * loss_boundary1
        loss_sem_color = stage1_weight * loss_sem_color0 + stage2_weight * loss_sem_color1
        loss_sem_tree = stage1_weight * loss_sem_tree0 + stage2_weight * loss_sem_tree1

        loss_total = (
            self.lambda_mask * (stage1_weight * loss_mask_aux + stage2_weight * loss_mask_final) +
            self.lambda_aux_mask * loss_mask_aux +
            self.lambda_point_mask * loss_point_mask +
            self.lambda_sem * loss_sem +
            self.lambda_boundary * loss_boundary +
            self.lambda_sem_color * loss_sem_color +
            self.lambda_sem_tree * loss_sem_tree
        )

        return {
            "loss_total": loss_total,
            "loss_mask_final": loss_mask_final,
            "loss_mask_aux": loss_mask_aux,
            "loss_point_mask": loss_point_mask,
            "loss_sem": loss_sem,
            "loss_boundary": loss_boundary,
            "loss_sem_color": loss_sem_color,
            "loss_sem_tree": loss_sem_tree,
            "stage2_weight": P1.new_tensor(stage2_weight),
        }
