"""
config.py
---------
Central configuration for the Rice Disease Classifier app.
Edit values here (image size, class names, checkpoint path, etc.)
instead of hunting through the rest of the codebase.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Default checkpoint filename expected inside the weights/ folder.
# Drop your trained .pth file there (or override the path from the app's
# sidebar at runtime).
DEFAULT_CHECKPOINT_NAME = "ProtoTypeEfficentNetB0CBAMV2Extra.pth"
DEFAULT_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, DEFAULT_CHECKPOINT_NAME)

# ---------------------------------------------------------------------------
# Image / model settings (must match training config)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
NUM_CLASSES = 4

# Order MUST match the ImageFolder class_to_idx used during training
# (i.e. alphabetical order of the training subfolders). Update this list to
# match your dataset's actual class folder names.
CLASS_NAMES = [
    "BrownSpot",
    "Healthy",
    "LeafBlast",
    "LeafBlight",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# CBAM settings (must match training config)
# ---------------------------------------------------------------------------
CBAM_CHANNELS = 1280  # EfficientNet-B0 final feature channel count
CBAM_REDUCTION = 16
CBAM_SPATIAL_KERNEL = 7

# ---------------------------------------------------------------------------
# Grad-CAM settings
# ---------------------------------------------------------------------------
GRADCAM_ALPHA = 0.5  # overlay transparency (kept for reference; grad-cam lib
                      # uses its own default blending inside show_cam_on_image)
# ---------------------------------------------------------------------------
# UI text (Indonesian) — short description + icon per class, shown on the
# landing/hero section of the app. Edit freely to match your actual dataset
# classes/descriptions.
# ---------------------------------------------------------------------------
DISEASE_INFO = {
    "BrownSpot": {
        "icon": "🟤",
        "label": "Bercak Coklat",
        "desc": "Muncul sebagai bercak coklat kecil berbentuk oval pada daun, dapat menurunkan hasil panen jika tidak ditangani.",
    },
    "Healthy": {
        "icon": "✅",
        "label": "Sehat",
        "desc": "Daun padi dalam kondisi baik tanpa tanda-tanda infeksi atau kerusakan penyakit.",
    },
    "LeafBlast": {
        "icon": "🔥",
        "label": "Blas Daun",
        "desc": "Lesi berbentuk belah ketupat dengan pusat abu-abu, salah satu penyakit paling merusak pada tanaman padi.",
    },
    "LeafBlight": {
        "icon": "🍂",
        "label": "Hawar Daun",
        "desc": "Menyebabkan daun menguning dan mengering dari ujung, dapat menyebar cepat pada kondisi lembap.",
    },
}