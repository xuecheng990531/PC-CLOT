"""Total training loss — point + transport + boundary only. No GT mask."""

import torch
import torch.nn as nn
from .pseudo_mask_losses import binary_bce_dice_loss
from .point_losses import point_mask_loss
from .boundary_affinity_loss import boundary_affinity_loss


class PCPointCLOTLoss(nn.Module):
    """PC-PointCLOT training loss (no GT mask).

    L_total = λ_point * L_point + λ_transport * L_transport + λ_boundary * L_boundary

    - L_point:      BCE at clicked point positions on final mask
    - L_transport:  BCE+Dice between each P^t and detach(T^t_fg)
    - L_boundary:   boundary affinity from transport neighborhood consistency
    """

    def __init__(
        self,
        lambda_point=1.0,
        lambda_transport=1.0,
        lambda_boundary=0.2,
        transport_stage_weights=None,
        eps=1e-6,
    ):
        super().__init__()
        self.lambda_point = lambda_point
        self.lambda_transport = lambda_transport
        self.lambda_boundary = lambda_boundary
        self.transport_stage_weights = transport_stage_weights or [0.4, 1.0]
        self.eps = eps

    def forward(self, outputs, point_maps, gt_mask=None):
        if gt_mask is not None:
            raise NotImplementedError(
                "GT mask is not used for training. "
                "All supervision comes from point_maps and pseudo-masks."
            )

        history = outputs["history"]
        P_list = history["P"]  # [P0, P1, ..., PK]  (K+1 items)
        B_list = history["B"]  # [B0, B1, ..., BK]
        T_list = history["T"]  # [T0, T1, ..., TK-1]  (K items)
        K = len(T_list)

        P_final = P_list[-1]

        # --- L_point: BCE at clicked points on EVERY stage prediction ---
        loss_point = 0.0
        for P_t in P_list:
            loss_point += point_mask_loss(P_t, point_maps)

        # --- L_transport: stage-wise BCE+Dice ---
        loss_transport = 0.0
        for t in range(K):
            P_t = P_list[t]
            target = T_list[t][:, 1:2].detach()
            w = self.transport_stage_weights[t] if t < len(self.transport_stage_weights) else 1.0
            loss_transport += w * binary_bce_dice_loss(P_t, target, self.eps)
        # also supervise final prediction with final transport
        loss_transport += binary_bce_dice_loss(P_final, T_list[-1][:, 1:2].detach(), self.eps)

        # --- L_boundary: stage-wise boundary affinity ---
        loss_boundary = 0.0
        for t in range(K):
            B_t = B_list[t]
            T_t = T_list[t].detach()
            loss_boundary += boundary_affinity_loss(B_t, T_t)

        # --- Total ---
        loss_total = (
            self.lambda_point * loss_point +
            self.lambda_transport * loss_transport +
            self.lambda_boundary * loss_boundary
        )

        return {
            "loss_total": loss_total,
            "loss_point": loss_point,
            "loss_transport": loss_transport,
            "loss_boundary": loss_boundary,
        }
