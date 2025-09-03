# models/memseg.py
# v1 - 2025-08-01, Forked from Memseg
# v2 - 2025-08-05, add ASFF option
# v3 - 2025-09-03, ASFF output channels resolution fix(expanded)
import torch.nn as nn
from .decoder import Decoder
from .msff import MSFF
from .asff import ASFF


class MemSeg(nn.Module):
    def __init__(self, memory_bank, feature_extractor, feature_channels, use_asff=False):
        super(MemSeg, self).__init__()

        self.memory_bank = memory_bank
        self.feature_extractor = feature_extractor

        f_in_channels = feature_channels[0]
        if use_asff:
            self.fusion = ASFF(out_channels=[64,128,256])
            fusion_channels = [f_in_channels, 256, 128, 64]
        else:
            self.fusion = MSFF()
            fusion_channels = [f_in_channels, 64, 128, 256]
        self.decoder = Decoder(fusion_channels=fusion_channels)

    def forward(self, inputs):
        # extract features
        features = self.feature_extractor(inputs)
        f_in = features[0]
        f_out = features[-1]
        f_ii = features[1:-1]

        # extract concatenated information(CI)
        concat_features = self.memory_bank.select(features = f_ii)

        # Feature Fusion Module (MSFF or ASFF)
        fusion_outputs = self.fusion(features = concat_features)

        # decoder
        predicted_mask = self.decoder(
            encoder_output  = f_out,
            concat_features = [f_in] + fusion_outputs
        )

        return predicted_mask
