from typing import Any

import torch
from torch import nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock3D, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x

class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(DownSampleBlock, self).__init__()
        self.enc_block = ConvBlock3D(in_channels, out_channels)
        self.down_block = nn.Conv3d(out_channels, out_channels, kernel_size=stride, stride=stride)

    def forward(self, x: torch.Tensor) -> tuple[Any, Any]:
        x = self.enc_block(x)
        skip = x
        x = self.down_block(x)
        return x, skip

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 2, 2)):
        super(UpSampleBlock, self).__init__()
        self.up_block = nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False)
        self.dec_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            ConvBlock3D(out_channels, out_channels)
        )

    def forward(self, x, skip = None):
        x = self.up_block(x)

        if skip is not None:
            prev_channels = x.shape[1]
            x = torch.cat([x, skip], dim=1)
            x = nn.Conv3d(x.shape[1], prev_channels, kernel_size=1).to(x.device)(x)

        x = self.dec_block(x)
        return x


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, levels: int = 3, preserve_time: bool = True,
                 use_skip_connections: bool = False):
        super(Autoencoder3D, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.stride = (1, 2, 2) if preserve_time else (2, 2, 2)

        # --- Encoder ---
        enc_blocks = []
        ch = in_channels
        chs = []
        for i in range(levels):
            out_ch = base_channels * (2 ** i)
            enc_blocks.append(DownSampleBlock(ch, out_ch, stride=self.stride))
            ch = out_ch
            chs.append(out_ch)

        self.enc_blocks = nn.ModuleList(enc_blocks)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock3D(ch, ch)

        # --- Decoder ---
        dec_blocks = []
        for i in reversed(range(levels)):
            in_ch = base_channels * (2 ** i)
            dec_blocks.append(UpSampleBlock(ch, in_ch, stride=self.stride))
            ch = in_ch

        self.dec_blocks = nn.ModuleList(dec_blocks)

        # --- Output block ---
        self.out_conv = nn.Conv3d(ch, in_channels, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: To include skip connections
        skips = []

        for enc_block in self.enc_blocks:
            x, skip = enc_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for out_block, skip in zip(self.dec_blocks, reversed(skips)):
            x = out_block(x, skip if self.use_skip_connections else None)

        y = self.out_conv(x)
        y = self.out_act(y)
        return y

