"""Prapemrosesan gambar, mengikuti transformasi validasi saat pelatihan."""

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from config.settings import IMAGE_SIZE, NORM_MEAN, NORM_STD

# Transformasi ini harus sama persis dengan `test_transform` pada notebook:
# resize -> tensor -> normalisasi. Tidak ada augmentasi acak.
transform_inferensi = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


def buka_gambar(berkas) -> Image.Image:
    """Membuka unggahan menjadi gambar RGB dengan orientasi yang benar."""
    gambar = Image.open(berkas)
    # Foto dari ponsel sering menyimpan orientasi di metadata EXIF.
    gambar = ImageOps.exif_transpose(gambar)
    return gambar.convert("RGB")


def siapkan_tensor(gambar: Image.Image) -> torch.Tensor:
    """Mengubah gambar PIL menjadi tensor batch berukuran (1, 3, 224, 224)."""
    return transform_inferensi(gambar).unsqueeze(0)


def gambar_dasar_untuk_overlay(gambar: Image.Image) -> np.ndarray:
    """Versi gambar berukuran 224x224 dalam rentang [0, 1] untuk overlay.

    Dipakai sebagai lapisan bawah visualisasi Grad-CAM. Sengaja diambil
    dari gambar asli, bukan dari tensor yang sudah dinormalisasi, agar
    warnanya tetap sesuai aslinya.
    """
    kecil = gambar.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(kecil, dtype=np.float32) / 255.0
