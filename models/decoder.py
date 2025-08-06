import torch
import torch.nn as nn


class UpConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpConvBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

    def forward(self, x):
        return self.blk(x)



class Decoder(nn.Module):
    def __init__(self,
                 encoder_out_channels=512,
                 fusion_channels=[128, 128, 128, 128],
                 decoder_channels=[256, 128, 64, 48],
                 final_out_channels=2):
        """
        encoder_out_channels: encoder_output의 채널 수 (ex. 512)
        fusion_channels: [f3, f2, f1, f0]의 채널 수 (ASFF: [128,128,128,128], MSFF: [256,256,256,64] 등)
        decoder_channels: 업샘플링 후 각 단계의 출력 채널 수 [up3, up2, up1, up0]
        final_out_channels: 최종 출력 채널 수 (ex. 2)
        """
        super(Decoder, self).__init__()

        # f0, f1, f2, f3 = concat_features
        f0_c, f1_c, f2_c, f3_c = fusion_channels
        up3_c, up2_c, up1_c, up0_c = decoder_channels

        self.conv = nn.Conv2d(f0_c, up0_c, kernel_size=3, stride=1, padding=1)

        self.upconv3 = UpConvBlock(encoder_out_channels, up3_c)         # encoder_output -> up3_c
        self.upconv2 = UpConvBlock(up3_c + f3_c, up2_c)                # x_up3 + f2 -> up2_c
        self.upconv1 = UpConvBlock(up2_c + f2_c, up1_c)                # x_up2 + f1 -> up1_c
        self.upconv0 = UpConvBlock(up1_c + f1_c, up0_c)                # x_up1 + f0 -> up0_c
        self.upconv2mask = UpConvBlock(up0_c + up0_c, up0_c)           # x_up0 + f0(conv) -> up0_c

        self.final_conv = nn.Conv2d(up0_c, final_out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, encoder_output, concat_features):
        # concat_features = [level0, level1, level2, level3]
        f0, f1, f2, f3 = concat_features
        
        # 512 x 8 x 8 -> 512 x 16 x 16
        x_up3 = self.upconv3(encoder_output)
        if x_up3.shape[-2:] != f3.shape[-2:]:
            f3 = nn.functional.interpolate(f3, size=x_up3.shape[-2:], mode='bilinear', align_corners=False)
        x_up3 = torch.cat([x_up3, f3], dim=1)  

        # 512 x 16 x 16 -> 256 x 32 x 32
        x_up2 = self.upconv2(x_up3)
        if x_up2.shape[-2:] != f2.shape[-2:]:
            f2 = nn.functional.interpolate(f2, size=x_up2.shape[-2:], mode='bilinear', align_corners=False)
        x_up2 = torch.cat([x_up2, f2], dim=1)  

        # 256 x 32 x 32 -> 128 x 64 x 64
        x_up1 = self.upconv1(x_up2)
        if x_up1.shape[-2:] != f1.shape[-2:]:
            f1 = nn.functional.interpolate(f1, size=x_up1.shape[-2:], mode='bilinear', align_corners=False)
        x_up1 = torch.cat([x_up1, f1], dim=1)  

        # 128 x 64 x 64 -> 96 x 128 x 128
        x_up0 = self.upconv0(x_up1)
        f0 = self.conv(f0)
        x_up2mask = torch.cat([x_up0, f0], dim=1)  

        # 96 x 128 x 128 -> 48 x 256 x 256
        x_mask = self.upconv2mask(x_up2mask)  
        
        # 48 x 256 x 256 -> 1 x 256 x 256
        x_mask = self.final_conv(x_mask)  
        
        return x_mask