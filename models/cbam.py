"""
cbam.py
-------
Convolutional Block Attention Module (CBAM): channel attention followed
by spatial attention. Ported directly from the original training notebook.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        # Mengubah setiap feature map menjadi nilai 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP menggunakan convolution 1 x 1
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Informasi rata-rata dari setiap channel
        avg_out = self.mlp(self.avg_pool(x))

        # Informasi nilai maksimum dari setiap channel
        max_out = self.mlp(self.max_pool(x))

        # Menggabungkan keduanya untuk menghasilkan bobot channel
        attention = self.sigmoid(avg_out + max_out)

        # Mengalikan feature map dengan bobot attention
        return x * attention


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        if kernel_size not in (3, 7):
            raise ValueError("kernel_size harus bernilai 3 atau 7.")

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Rata-rata sepanjang dimensi channel
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Nilai maksimum sepanjang dimensi channel
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Menggabungkan avg pooling dan max pooling
        attention = torch.cat([avg_out, max_out], dim=1)

        # Membentuk spatial attention map
        attention = self.conv(attention)
        attention = self.sigmoid(attention)

        # Mengalikan feature map dengan spatial attention
        return x * attention


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=reduction,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
        )

    def forward(self, x):
        # Tahap pertama: channel attention
        x = self.channel_attention(x)

        # Tahap kedua: spatial attention
        x = self.spatial_attention(x)

        return x
