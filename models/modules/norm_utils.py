import torch.nn as nn


def get_groupnorm(channels, groups=8):
    """Get GroupNorm with automatic group adjustment.

    If channels is divisible by groups, use GroupNorm(groups, channels).
    Otherwise, use GroupNorm(1, channels).
    """
    if channels % groups == 0:
        return nn.GroupNorm(groups, channels)
    else:
        return nn.GroupNorm(1, channels)
