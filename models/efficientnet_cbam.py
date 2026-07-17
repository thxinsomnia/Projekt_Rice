"""
efficientnet_cbam.py
---------------------
EfficientNet-B0 backbone with a CBAM attention block inserted before the
global average pooling / classifier head. Same architecture used to train
the checkpoint referenced in config.py.
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .cbam import CBAM


class EfficientNetB0_CBAM(nn.Module):
    def __init__(self, num_classes, cbam_channels=1280, cbam_reduction=16,
                 cbam_spatial_kernel=7, pretrained=True):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_b0(weights=weights)

        self.features = base_model.features

        self.cbam = CBAM(
            channels=cbam_channels,
            reduction=cbam_reduction,
            spatial_kernel_size=cbam_spatial_kernel,
        )

        self.avgpool = base_model.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(cbam_channels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(num_classes, checkpoint_path=None, device="cpu",
                 cbam_channels=1280, cbam_reduction=16, cbam_spatial_kernel=7):
    """
    Build the model and (optionally) load trained weights.

    When loading a checkpoint we skip downloading the ImageNet-pretrained
    backbone weights (pretrained=False) since they'll be immediately
    overwritten anyway -- this makes app startup faster and works offline.
    """
    pretrained = checkpoint_path is None

    model = EfficientNetB0_CBAM(
        num_classes=num_classes,
        cbam_channels=cbam_channels,
        cbam_reduction=cbam_reduction,
        cbam_spatial_kernel=cbam_spatial_kernel,
        pretrained=pretrained,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model
