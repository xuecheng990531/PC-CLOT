"""CLOT-style cross-attention refinement: prototype-to-pixel feature feedback."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.norm_utils import get_groupnorm


class CrossAttentionRefinement(nn.Module):
    """Refine feature map via cross-attention from learned prototypes (CLOT-style).

    Input:
        feature:    [B, C, H, W]
        prototypes: [B, K, C]   K is configurable

    Output:
        refined_feature: [B, C, H, W]  (same shape as input)
    """

    def __init__(
        self,
        channels,
        num_prototypes=3,
        temperature=1.0,
        use_projection=True,
        residual_scale=1.0,
        norm="gn",
    ):
        super().__init__()
        self.channels = channels
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.use_projection = use_projection
        self.residual_scale = residual_scale

        if use_projection:
            self.q_proj = nn.Linear(channels, channels)
            self.k_proj = nn.Linear(channels, channels)
            self.v_proj = nn.Linear(channels, channels)

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            get_groupnorm(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, prototypes):
        """
        Args:
            feature:    [B, C, H, W]
            prototypes: [B, K, C]

        Returns:
            refined_feature: [B, C, H, W]
        """
        B, C, H, W = feature.shape
        N = H * W

        X = feature.reshape(B, C, N).transpose(1, 2)  # [B, N, C]

        if self.use_projection:
            Q = self.q_proj(X)  # [B, N, C]
            K = self.k_proj(prototypes)  # [B, K, C]
            V = self.v_proj(prototypes)  # [B, K, C]
        else:
            Q = X
            K = prototypes
            V = prototypes

        scale = self.temperature * (C ** 0.5)
        attn_logits = torch.bmm(Q, K.transpose(1, 2)) / scale  # [B, N, K]
        attn = F.softmax(attn_logits, dim=-1)  # [B, N, K]

        context = torch.bmm(attn, V)  # [B, N, C]
        context = context.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        refined = feature + self.residual_scale * self.out_proj(context)
        return refined
