"""
gradcam_utils.py
-----------------
Thin wrapper around pytorch_grad_cam to build a GradCAM object for the
model's last convolutional block and produce an overlay image ready to
show in the Streamlit UI.
"""

import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .preprocessing import denormalize_image


def get_gradcam(model):
    """
    Build a GradCAM object targeting the last conv block of the
    EfficientNet feature extractor (model.features[-1]).
    """
    target_layers = [model.features[-1]]
    return GradCAM(model=model, target_layers=target_layers)


def generate_gradcam_overlay(cam: GradCAM, input_tensor, target_category=None):
    """
    Run Grad-CAM on a single preprocessed image tensor (1, 3, H, W) and
    return an RGB uint8 overlay image (H, W, 3).

    target_category=None lets pytorch_grad_cam use the model's own
    predicted class, matching the notebook's `targets=None` behaviour.
    """
    targets = None
    if target_category is not None:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_category)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0]  # first (only) image in the batch

    original_image = denormalize_image(input_tensor[0])

    overlay = show_cam_on_image(
        original_image,
        grayscale_cam,
        use_rgb=True,
    )

    return overlay, original_image
