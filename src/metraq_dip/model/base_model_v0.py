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


class Autoencoder3D(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, levels: int = 3, preserve_time: bool = True,
                 use_skip_connections: bool = False):
        super(Autoencoder3D, self).__init__()
        self.use_skip_connections = use_skip_connections
        self.stride = (1, 2, 2) if preserve_time else (2, 2, 2)

        # --- Encoder ---
        enc_blocks = []
        pools = []
        ch = in_channels
        chs = []
        for i in range(levels):
            out_ch = base_channels * (2 ** i)
            enc_blocks.append(ConvBlock3D(ch, out_ch))
            pools.append(nn.Conv3d(out_ch, out_ch, kernel_size=self.stride, stride=self.stride))
            ch = out_ch
            chs.append(out_ch)

        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.downs = nn.ModuleList(pools)
        self.bottleneck = ConvBlock3D(ch, ch)

        # --- Decoder ---
        dec_blocks = []
        ups = []
        for i in reversed(range(levels)):
            in_ch = base_channels * (2 ** i)
            ups.append(self.stride)
            dec_blocks.append(nn.Sequential(
                nn.Conv3d(ch, in_ch, kernel_size=1),
                nn.ReLU(inplace=True),
                ConvBlock3D(in_ch, in_ch),
            ))
            ch = in_ch

        self.dec_blocks = nn.ModuleList(dec_blocks)
        self.up_scales = ups
        self.out_conv = nn.Conv3d(ch, in_channels, kernel_size=1)
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: To include skip connections
        skips = []

        for block, down in zip(self.enc_blocks, self.downs):
            x = block(x)
            if self.use_skip_connections:
                skips.append(x)
            x = down(x)

        x = self.bottleneck(x)
        for scale, block, skip in zip(self.up_scales, self.dec_blocks, reversed(skips)):
            x = nn.functional.interpolate(x, scale_factor=scale, mode='trilinear', align_corners=False)
            if self.use_skip_connections:
                prev_channels = x.shape[1]
                x = torch.cat([x, skip], dim=1)
                x = nn.Conv3d(x.shape[1], prev_channels, kernel_size=1).to(x.device)(x)
            x = block(x)

        y = self.out_conv(x)
        y = self.out_act(y)
        return y

