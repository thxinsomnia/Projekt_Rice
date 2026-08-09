"""Gradient-weighted Class Activation Mapping (Selvaraju et al., 2017).

Ditulis langsung memakai hook PyTorch, tanpa pustaka pihak ketiga, agar
lapisan sasaran bisa ditentukan bebas dan aplikasi punya dependensi lebih
sedikit.

Lapisan sasaran yang dipakai adalah keluaran modul CBAM, bukan
`features[-1]`. Pilihan ini disengaja: peta aktivasi yang divisualisasikan
menjadi hasil SETELAH attention diterapkan, sehingga betul-betul
menggambarkan area yang ditonjolkan oleh CBAM.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config.settings import IMAGE_SIZE


class GradCAM:
    """Menghasilkan peta aktivasi kelas dari satu lapisan konvolusi."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = []

    def _simpan_aktivasi(self, module, masukan, keluaran):
        self.activations = keluaran.detach()

    def _simpan_gradien(self, module, grad_masukan, grad_keluaran):
        self.gradients = grad_keluaran[0].detach()

    def __enter__(self) -> "GradCAM":
        self._handles = [
            self.target_layer.register_forward_hook(self._simpan_aktivasi),
            self.target_layer.register_full_backward_hook(self._simpan_gradien),
        ]
        return self

    def __exit__(self, *args) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self.activations = None
        self.gradients = None

    def hasilkan(
        self,
        tensor_masukan: torch.Tensor,
        indeks_kelas: Optional[int] = None,
    ) -> np.ndarray:
        """Menghasilkan heatmap [0, 1] berukuran IMAGE_SIZE x IMAGE_SIZE.

        Bila `indeks_kelas` tidak diisi, Grad-CAM diarahkan ke kelas yang
        diprediksi model.
        """
        self.model.zero_grad(set_to_none=True)

        # Input perlu melacak gradien. Tanpa ini, backward akan gagal
        # ketika seluruh parameter model dibekukan.
        tensor_masukan = tensor_masukan.clone().requires_grad_(True)

        logits = self.model(tensor_masukan)

        if indeks_kelas is None:
            indeks_kelas = int(logits.argmax(dim=1).item())

        skor = logits[0, indeks_kelas]
        skor.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError(
                "Aktivasi atau gradien tidak tertangkap. "
                "Pastikan Grad-CAM dipakai di dalam blok `with`."
            )

        # Bobot tiap channel = rata-rata gradien secara spasial.
        bobot = self.gradients.mean(dim=(2, 3), keepdim=True)

        cam = (bobot * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        rentang = cam.max() - cam.min()
        if rentang < 1e-8:
            # Aktivasi seragam, misalnya bila prediksi sangat tidak yakin.
            return np.zeros_like(cam)

        return (cam - cam.min()) / rentang


def peta_warna_jet(heatmap: np.ndarray) -> np.ndarray:
    """Mengubah heatmap [0, 1] menjadi RGB dengan skema biru-hijau-merah.

    Ditulis manual agar matplotlib tidak perlu dimuat hanya untuk warna.
    Biru menandakan pengaruh kecil, merah pengaruh besar.
    """
    x = np.clip(heatmap, 0.0, 1.0)

    merah = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    hijau = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    biru = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    return np.stack([merah, hijau, biru], axis=-1)


def tumpuk_pada_gambar(
    gambar_dasar: np.ndarray,
    heatmap: np.ndarray,
    opasitas: float = 0.5,
) -> np.ndarray:
    """Menumpuk heatmap di atas gambar asli.

    Parameter
    ---------
    gambar_dasar : np.ndarray
        Gambar RGB [0, 1] berukuran (224, 224, 3).
    heatmap : np.ndarray
        Peta aktivasi [0, 1] berukuran (224, 224).
    opasitas : float
        0.0 menampilkan gambar asli sepenuhnya,
        1.0 menampilkan heatmap sepenuhnya.
    """
    warna = peta_warna_jet(heatmap)
    hasil = (1.0 - opasitas) * gambar_dasar + opasitas * warna
    return np.clip(hasil * 255.0, 0, 255).astype(np.uint8)
