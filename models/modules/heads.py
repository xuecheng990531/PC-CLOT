import torch.nn as nn
from .norm_utils import get_groupnorm


class SemanticHead(nn.Module):
    """Semantic segmentation head: predicts foreground vs background."""

    def __init__(self, in_channels=64, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = get_groupnorm(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, F0):
        x = self.conv(F0)
        x = self.norm(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class BoundaryHead(nn.Module):
    """Boundary prediction head."""

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = get_groupnorm(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, F0):
        x = self.conv(F0)
        x = self.norm(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class MaskHead(nn.Module):
    """Mask prediction head."""

    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm = get_groupnorm(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, F0):
        x = self.conv(F0)
        x = self.norm(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x
