import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import (
    DoubleConv,
    DownBlock,
    UpBlock,
    PCSkipFusion,
    BoundaryHead,
    MaskHead,
)


class PCUNet(nn.Module):
    """Point-guided U-Net backbone with Predictor-Corrector enhanced skip fusion.

    Input:
        - image:      [B, 3, H, W]
        - point_maps: [B, 2, H, W]  (ch0=foreground, ch1=background, mandatory)

    Output:
        dict with keys: F0, mask_logits, boundary_logits,
                       encoder_features, enhanced_skips
    """

    def __init__(
        self,
        input_channels=5,
        encoder_channels=None,
        memory_channels=32,
        delta=1.0,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 128, 256, 512, 1024]

        self.input_channels = input_channels
        self.encoder_channels = encoder_channels
        self.memory_channels = memory_channels
        self.delta = delta

        self._build_encoder(input_channels, encoder_channels)
        self._build_pc_skip_fusion()
        self._build_decoder(encoder_channels)
        self._build_heads()

    def _build_encoder(self, input_channels, encoder_channels):
        self.init_conv = DoubleConv(input_channels, encoder_channels[0])
        self.enc1 = DownBlock(encoder_channels[0], encoder_channels[1])
        self.enc2 = DownBlock(encoder_channels[1], encoder_channels[2])
        self.enc3 = DownBlock(encoder_channels[2], encoder_channels[3])
        self.enc4 = DownBlock(encoder_channels[3], encoder_channels[4])
        self.bottleneck = DoubleConv(encoder_channels[4], encoder_channels[4])

    def _build_pc_skip_fusion(self):
        skip_channels = self.encoder_channels[:4]
        self.pc_skip_fusion = PCSkipFusion(
            input_channels=skip_channels,
            memory_channels=self.memory_channels,
            delta=self.delta,
        )

    def _build_decoder(self, encoder_channels):
        self.dec4 = UpBlock(encoder_channels[4], encoder_channels[3], encoder_channels[3])
        self.dec3 = UpBlock(encoder_channels[3], encoder_channels[2], encoder_channels[2])
        self.dec2 = UpBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1])
        self.dec1 = UpBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0])

    def _build_heads(self):
        self.mask_head = MaskHead(in_channels=self.encoder_channels[0])
        self.boundary_head = BoundaryHead(in_channels=self.encoder_channels[0])

    def encode(self, x):
        x0 = self.init_conv(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        return [x0, x1, x2, x3, x4, x5]

    def decode(self, x5, enhanced_skips):
        s1, s2, s3, s4 = enhanced_skips
        d4 = self.dec4(x5, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        return d1

    def predict_from_feature(self, feature):
        """Shared heads: mask + boundary from any feature map.

        Args:
            feature: [B, C, H, W]

        Returns:
            mask_logits:     [B, 1, H, W]
            boundary_logits: [B, 1, H, W]
        """
        return self.mask_head(feature), self.boundary_head(feature)

    def forward(self, image, point_maps):
        if point_maps is None:
            raise ValueError("point_maps is mandatory and cannot be None")

        x = torch.cat([image, point_maps], dim=1)
        encoder_features = self.encode(x)
        x0, x1, x2, x3, x4, x5 = encoder_features

        enhanced_skips = self.pc_skip_fusion([x0, x1, x2, x3])
        F0 = self.decode(x5, enhanced_skips)

        P0, B0 = self.predict_from_feature(F0)

        return {
            "F0": F0,
            "mask_logits": P0,
            "boundary_logits": B0,
            "encoder_features": [x0, x1, x2, x3, x4, x5],
            "enhanced_skips": enhanced_skips,
        }

    def reset_pc_memory(self):
        self.pc_skip_fusion.reset_memory()
