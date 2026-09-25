import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Lightweight CBAM for fused high-level features."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class PointGuidedInteraction(nn.Module):
    """Inject image+point guidance into fused features by residual gating."""

    def __init__(self, feature_channels, guide_channels=32, residual_scale=0.5):
        super().__init__()
        self.residual_scale = residual_scale
        self.guide = nn.Sequential(
            nn.Conv2d(5, guide_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, guide_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(guide_channels, feature_channels, kernel_size=1, bias=False),
        )

    def forward(self, feature, image, point_maps):
        image_small = F.interpolate(image, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        point_small = F.interpolate(point_maps, size=feature.shape[-2:], mode="nearest")
        gate = torch.sigmoid(self.guide(torch.cat([image_small, point_small], dim=1)))
        return feature * (1.0 + self.residual_scale * gate)
