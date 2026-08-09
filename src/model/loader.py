"""Pemuatan bobot model, baik dari berkas bawaan maupun unggahan pengguna."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st
import torch

from config.settings import CLASS_NAMES, DEFAULT_MODEL_PATH, NUM_CLASSES
from src.model.architecture import EfficientNetB0_CBAM


@dataclass
class LoadedModel:
    """Model siap pakai beserta keterangan asal bobotnya."""

    model: torch.nn.Module
    device: torch.device
    sumber: str
    jumlah_parameter: int
    ukuran_mb: float


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _hitung_ukuran_mb(model: torch.nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024**2)


def _bangun_dari_state_dict(state_dict: dict, sumber: str) -> LoadedModel:
    device = get_device()

    model = EfficientNetB0_CBAM(num_classes=NUM_CLASSES, pretrained=False)

    # Checkpoint kadang dibungkus, misalnya {"state_dict": ...} atau
    # {"model_state_dict": ...}. Ambil isinya bila terdeteksi.
    for kunci in ("state_dict", "model_state_dict"):
        if kunci in state_dict and isinstance(state_dict[kunci], dict):
            state_dict = state_dict[kunci]
            break

    # Buang awalan "module." bila model dilatih memakai DataParallel.
    state_dict = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        raise ValueError(
            "Struktur bobot tidak cocok dengan arsitektur EfficientNet-B0 + CBAM.\n"
            f"Lapisan yang tidak terisi: {len(missing)}. "
            f"Lapisan tak dikenali: {len(unexpected)}.\n"
            "Pastikan berkas berasal dari pelatihan arsitektur yang sama "
            f"dengan {NUM_CLASSES} kelas."
        )

    model.to(device)
    model.eval()

    return LoadedModel(
        model=model,
        device=device,
        sumber=sumber,
        jumlah_parameter=sum(p.numel() for p in model.parameters()),
        ukuran_mb=_hitung_ukuran_mb(model),
    )


@st.cache_resource(show_spinner=False)
def muat_model_bawaan(path_str: str, _mtime: float) -> LoadedModel:
    """Memuat model bawaan dari folder models/.

    Parameter `_mtime` ikut menentukan kunci cache, sehingga berkas yang
    diganti akan otomatis dimuat ulang tanpa perlu restart.
    """
    path = Path(path_str)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return _bangun_dari_state_dict(state_dict, sumber=path.name)


@st.cache_resource(show_spinner=False)
def muat_model_unggahan(berkas_bytes: bytes, nama_berkas: str) -> LoadedModel:
    """Memuat model dari berkas .pth yang diunggah pengguna."""
    state_dict = torch.load(
        io.BytesIO(berkas_bytes),
        map_location="cpu",
        weights_only=True,
    )
    return _bangun_dari_state_dict(state_dict, sumber=nama_berkas)


def sidik_jari(data: bytes) -> str:
    """Ringkasan pendek isi berkas, dipakai sebagai penanda di antarmuka."""
    return hashlib.sha256(data).hexdigest()[:8]


def model_bawaan_tersedia() -> bool:
    return DEFAULT_MODEL_PATH.exists()


def ambil_model(
    berkas_unggahan: Optional[object] = None,
) -> tuple[Optional[LoadedModel], Optional[str]]:
    """Mengembalikan (model, pesan_kesalahan).

    Bila `berkas_unggahan` diisi, bobot itu yang dipakai. Bila tidak,
    aplikasi memakai model bawaan di folder models/.
    """
    try:
        if berkas_unggahan is not None:
            data = berkas_unggahan.getvalue()
            return muat_model_unggahan(data, berkas_unggahan.name), None

        if not model_bawaan_tersedia():
            return None, (
                f"Model bawaan belum ada. Salin berkas "
                f"`{DEFAULT_MODEL_PATH.name}` ke folder `models/`, "
                "atau unggah bobot lain lewat panel di samping."
            )

        return (
            muat_model_bawaan(
                str(DEFAULT_MODEL_PATH),
                DEFAULT_MODEL_PATH.stat().st_mtime,
            ),
            None,
        )

    except ValueError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001
        return None, f"Berkas bobot gagal dibaca: {exc}"


def daftar_kelas() -> list[str]:
    return list(CLASS_NAMES)
