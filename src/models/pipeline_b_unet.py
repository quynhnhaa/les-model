"""Pipeline B U-Net with different architecture and training strategy"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from src.models.layers import ConvBnRelu, UBlock, conv1x1, UBlockCbam, CBAM


class PipelineB_Unet(nn.Module):
    """Pipeline B U-Net with different architecture and training strategy.
    
    Key differences from Pipeline A:
    - Different width scaling
    - Different attention mechanisms
    - Different normalization strategies
    """
    name = "PipelineB_Unet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0.1,
                 **kwargs):
        super(PipelineB_Unet, self).__init__()
        # Different feature scaling for Pipeline B
        features = [width * 2 ** i for i in range(4)]
        print(f"Pipeline B features: {features}")

        self.deep_supervision = deep_supervision

        # Use CBAM attention in encoder for Pipeline B
        self.encoder1 = UBlockCbam(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)

        # Different bottom architecture
        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        # Additional processing in bottom
        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        # Decoder with attention
        self.decoder3 = UBlockCbam(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlockCbam(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlockCbam(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0] // 2, num_classes)

        # Deep supervision with different weights
        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()

    def _init_weights(self):
        # Different initialization strategy for Pipeline B
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)

        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder
        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        if self.deep_supervision:
            deeps = []
            for seg, deep in zip(
                    [bottom, bottom_2, up3, up2],
                    [self.deep_bottom, self.deep_bottom2, self.deep3, self.deep2]):
                deeps.append(deep(seg))
            return out, deeps

        return out


class PipelineB_EquiUnet(PipelineB_Unet):
    """Pipeline B EquiUnet variant"""
    name = "PipelineB_EquiUnet"

    def __init__(self, inplanes, num_classes, width, norm_layer=None, deep_supervision=False, dropout=0.1,
                 **kwargs):
        super(PipelineB_Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print(f"Pipeline B EquiUnet features: {features}")

        self.deep_supervision = deep_supervision

        # EquiUnet architecture with CBAM
        self.encoder1 = UBlockCbam(inplanes, features[0], features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlockCbam(features[0], features[1], features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlockCbam(features[1], features[2], features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlockCbam(features[2], features[3], features[3], norm_layer, dropout=dropout)

        self.bottom = UBlockCbam(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)

        self.bottom_2 = nn.Sequential(
            ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout),
            CBAM(features[2], norm_layer=norm_layer)
        )

        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlockCbam(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlockCbam(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlockCbam(features[0] * 2, features[0], features[0], norm_layer, dropout=dropout)

        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)

        self.outconv = conv1x1(features[0], num_classes)

        if self.deep_supervision:
            self.deep_bottom = nn.Sequential(
                conv1x1(features[3], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep_bottom2 = nn.Sequential(
                conv1x1(features[2], num_classes),
                nn.Upsample(scale_factor=8, mode="trilinear", align_corners=True))

            self.deep3 = nn.Sequential(
                conv1x1(features[1], num_classes),
                nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True))

            self.deep2 = nn.Sequential(
                conv1x1(features[0], num_classes),
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True))

        self._init_weights()
