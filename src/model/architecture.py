"""Arsitektur EfficientNet-B0 yang diintegrasikan dengan CBAM."""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from src.model.cbam import CBAM

FEATURE_CHANNELS = 1280


class EfficientNetB0_CBAM(nn.Module):
    """EfficientNet-B0 dengan modul CBAM sebelum lapisan klasifikasi.

    Alur: features -> cbam -> avgpool -> flatten -> classifier

    Parameter
    ---------
    num_classes : int
        Jumlah kelas keluaran.
    pretrained : bool
        Bila True, memuat bobot ImageNet. Untuk inferensi di aplikasi
        nilainya False karena seluruh bobot akan ditimpa oleh state_dict
        hasil pelatihan, sehingga tidak perlu mengunduh apa pun.
    """

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()

        base_model = efficientnet_b0(weights="DEFAULT" if pretrained else None)

        self.features = base_model.features
        self.cbam = CBAM(channels=FEATURE_CHANNELS)
        self.avgpool = base_model.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(FEATURE_CHANNELS, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EfficientNetB0_Baseline(nn.Module):
    """Varian tanpa CBAM, dipakai sebagai pembanding pada halaman Performa."""

    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()

        base_model = efficientnet_b0(weights="DEFAULT" if pretrained else None)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(FEATURE_CHANNELS, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
