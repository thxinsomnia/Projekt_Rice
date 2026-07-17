"""
preprocessing.py
----------------
Image preprocessing helpers for inference: the exact same resize +
normalize pipeline used on the validation/test set during training
(no augmentation), plus a de-normalize helper for display / Grad-CAM.
"""

import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32)
_STD = np.array(IMAGENET_STD, dtype=np.float32)


def get_inference_transform(image_size=IMAGE_SIZE):
    """Deterministic resize + normalize pipeline (mirrors test_transform)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(pil_image: Image.Image, image_size=IMAGE_SIZE):
    """
    Convert a PIL image (as uploaded by the user) into a normalized
    tensor of shape (1, 3, H, W) ready to feed to the model.
    """
    pil_image = pil_image.convert("RGB")
    transform = get_inference_transform(image_size)
    tensor = transform(pil_image)
    return tensor.unsqueeze(0)  # add batch dimension


def denormalize_image(image_tensor):
    """
    Reverse ImageNet normalization on a single (C, H, W) tensor and return
    an RGB float32 numpy array in the 0-1 range, suitable for Grad-CAM
    overlay or plt.imshow.
    """
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * _STD + _MEAN
    image = np.clip(image, 0, 1)
    return image.astype(np.float32)
