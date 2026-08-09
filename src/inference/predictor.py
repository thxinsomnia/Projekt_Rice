"""Menjalankan prediksi sekaligus menyiapkan bahan visualisasi Grad-CAM."""

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from config.settings import CLASS_NAMES
from src.inference.gradcam import GradCAM
from src.inference.preprocess import (
    gambar_dasar_untuk_overlay,
    siapkan_tensor,
)
from src.model.loader import LoadedModel


@dataclass
class HasilPrediksi:
    """Seluruh keluaran satu kali analisis."""

    kelas: str
    confidence: float
    probabilitas: dict[str, float] = field(default_factory=dict)
    heatmap: np.ndarray = None
    gambar_dasar: np.ndarray = None
    waktu_ms: float = 0.0

    @property
    def peringkat(self) -> list[tuple[str, float]]:
        """Seluruh kelas terurut dari probabilitas tertinggi."""
        return sorted(
            self.probabilitas.items(),
            key=lambda pasangan: pasangan[1],
            reverse=True,
        )

    @property
    def runner_up(self) -> tuple[str, float]:
        """Kelas dengan probabilitas tertinggi kedua."""
        return self.peringkat[1]

    @property
    def selisih_dengan_runner_up(self) -> float:
        return self.confidence - self.runner_up[1]


def prediksi(
    dimuat: LoadedModel,
    gambar: Image.Image,
    hitung_gradcam: bool = True,
) -> HasilPrediksi:
    """Menjalankan satu inferensi lengkap terhadap sebuah gambar."""
    model = dimuat.model
    device = dimuat.device

    tensor = siapkan_tensor(gambar).to(device)

    mulai = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probabilitas = torch.softmax(logits, dim=1)[0]
    if device.type == "cuda":
        torch.cuda.synchronize()
    waktu_ms = (time.perf_counter() - mulai) * 1000

    nilai = probabilitas.cpu().numpy()
    indeks_teratas = int(nilai.argmax())

    heatmap = None
    if hitung_gradcam:
        # Lapisan sasaran adalah keluaran CBAM, bukan blok konvolusi
        # terakhir, agar visualisasi mencerminkan hasil setelah attention.
        lapisan_sasaran = getattr(model, "cbam", None)
        if lapisan_sasaran is None:
            lapisan_sasaran = model.features[-1]

        with GradCAM(model, lapisan_sasaran) as cam:
            heatmap = cam.hasilkan(tensor, indeks_kelas=indeks_teratas)

    return HasilPrediksi(
        kelas=CLASS_NAMES[indeks_teratas],
        confidence=float(nilai[indeks_teratas]),
        probabilitas={
            nama: float(nilai[i]) for i, nama in enumerate(CLASS_NAMES)
        },
        heatmap=heatmap,
        gambar_dasar=gambar_dasar_untuk_overlay(gambar),
        waktu_ms=waktu_ms,
    )
