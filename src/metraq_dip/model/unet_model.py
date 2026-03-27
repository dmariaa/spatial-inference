from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int, int]):
        super().__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.downsample = nn.MaxPool3d(kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv_block(x)
        x = self.downsample(skip)
        return x, skip


class DecoderBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        learned_upsampling: bool = False,
    ):
        super().__init__()
        self.learned_upsampling = learned_upsampling

        if learned_upsampling:
            self.upscale = nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
            )
        else:
            self.upscale = nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False)
            self.upscale_projection = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.merge_block = ConvBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upscale(x)
        if not self.learned_upsampling:
            x = self.upscale_projection(x)

        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.merge_block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        levels: int = 3,
        preserve_time: bool = True,
        learned_upsampling: bool = False,
    ):
        super().__init__()

        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        if base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {base_channels}")

        self.learned_upsampling = learned_upsampling
        self.stride = (1, 2, 2) if preserve_time else (2, 2, 2)

        encoder_channels = [base_channels * (2 ** level) for level in range(levels)]
        bottleneck_channels = encoder_channels[-1] * 2

        encoder_blocks = []
        current_channels = in_channels
        for out_block_channels in encoder_channels:
            encoder_blocks.append(EncoderBlock3D(current_channels, out_block_channels, stride=self.stride))
            current_channels = out_block_channels
        self.encoder_blocks = nn.ModuleList(encoder_blocks)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(current_channels, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(bottleneck_channels),
            nn.ReLU(inplace=True),
            ConvBlock3D(bottleneck_channels, bottleneck_channels),
        )

        decoder_blocks = []
        current_channels = bottleneck_channels
        for skip_channels in reversed(encoder_channels):
            decoder_blocks.append(
                DecoderBlock3D(
                    in_channels=current_channels,
                    skip_channels=skip_channels,
                    out_channels=skip_channels,
                    stride=self.stride,
                    learned_upsampling=self.learned_upsampling,
                )
            )
            current_channels = skip_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        self.out_conv = nn.Conv3d(current_channels, out_channels, kernel_size=1)
        self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for encoder_block in self.encoder_blocks:
            x, skip = encoder_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder_block, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder_block(x, skip)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x


__all__ = ["UNet3D"]
