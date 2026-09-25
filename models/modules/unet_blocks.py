import torch
import torch.nn as nn
import torch.nn.functional as F
from .norm_utils import get_groupnorm


class DoubleConv(nn.Module):
    """Double 3x3 convolution block with GroupNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = get_groupnorm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = get_groupnorm(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block: MaxPool2d + DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block: bilinear upsample + concatenate skip + DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class EncoderBlock(nn.Module):
    """Encoder block: outputs both intermediate skip feature and downsampled feature."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.down = DownBlock(out_channels, out_channels * 2)

    def forward(self, x):
        skip = self.conv(x)
        out = self.down(skip)
        return skip, out


class DecoderBlock(nn.Module):
    """Decoder block: upsample + concatenate skip + DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = UpBlock(in_channels, skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x, skip)
        return x
