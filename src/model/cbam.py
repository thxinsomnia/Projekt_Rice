"""Convolutional Block Attention Module (Woo et al., 2018).

Implementasi disalin persis dari notebook pelatihan agar struktur
state_dict cocok saat bobot hasil pelatihan dimuat kembali.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Menentukan channel mana yang paling informatif ("apa")."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP diimplementasikan sebagai konvolusi 1x1.
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """Menentukan posisi mana yang paling informatif ("di mana")."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size harus bernilai 3 atau 7.")

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class CBAM(nn.Module):
    """Channel attention diikuti spatial attention secara berurutan."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
