# models/asff.py
# Adaptive Spatial Feature Fusion (ASFF) module for MemSeg
# Reference: https://arxiv.org/abs/1911.09516
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASFFBlock(nn.Module):
    """
    Adaptive Spatial Feature Fusion Block for a single scale.
    Args:
        level (int): 0 (highest), 1 (mid), 2 (lowest) scale.
        in_channels (list): List of input channels for each scale.
        out_channels (int): Output channels after fusion.
    """
    def __init__(self, level, in_channels, out_channels):
        super(ASFFBlock, self).__init__()
        self.level = level
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Project all inputs to the same number of channels
        self.inter_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1, 1, 0) for c in in_channels
        ])

        # Adjust spatial size for each input to match the target level
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # Learnable weights for each input
        self.weight_layers = nn.ModuleList([
            nn.Conv2d(out_channels, 1, 1, 1, 0) for _ in range(3)
        ])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        # features: list of 3 tensors from different scales
        # Project to same channel
        feats = [conv(f) for conv, f in zip(self.inter_convs, features)]

        # Resize to match target level
        if self.level == 0:
            # Highest scale, downsample others
            feats[1] = self.downsample(feats[1])
            feats[2] = self.downsample(self.downsample(feats[2]))
        elif self.level == 1:
            # Middle scale
            feats[0] = self.upsample(feats[0])
            feats[2] = self.downsample(feats[2])
        elif self.level == 2:
            # Lowest scale, upsample others
            feats[0] = self.upsample(self.upsample(feats[0]))
            feats[1] = self.upsample(feats[1])

        # Ensure all feats have the same spatial size before stacking
        target_size = feats[0].shape[-2:]
        feats = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) if f.shape[-2:] != target_size else f for f in feats]
        fused = torch.stack(feats, dim=1)  # (B, 3, C, H, W)

        # Compute weights
        weights = [layer(f) for layer, f in zip(self.weight_layers, feats)]
        weights = torch.cat(weights, dim=1)  # (B, 3, H, W)
        weights = self.softmax(weights)
        weights = weights.unsqueeze(2)  # (B, 3, 1, H, W)

        # Weighted sum
        out = (fused * weights).sum(dim=1)
        return out


class ASFF(nn.Module):
    """
    Adaptive Spatial Feature Fusion module for MemSeg.
    Args:
        in_channels (list): List of input channels for each scale [C1, C2, C3].
        out_channels (int): Output channels after fusion.
    """
    def __init__(self, in_channels=[128, 256, 512], out_channels=128):
        super(ASFF, self).__init__()
        self.block1 = ASFFBlock(level=0, in_channels=in_channels, out_channels=out_channels)
        self.block2 = ASFFBlock(level=1, in_channels=in_channels, out_channels=out_channels)
        self.block3 = ASFFBlock(level=2, in_channels=in_channels, out_channels=out_channels)

    def forward(self, features):
        # features: [f1, f2, f3] from different scales
        f1_out = self.block1(features)
        f2_out = self.block2(features)
        f3_out = self.block3(features)
        return [f1_out, f2_out, f3_out]
