"""Konfigurasi terpusat aplikasi deteksi penyakit daun padi."""

from pathlib import Path

# --- Path proyek ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"

DEFAULT_MODEL_PATH = MODEL_DIR / "EfficientNetREBORNv3.pth"
DISEASE_INFO_PATH = DATA_DIR / "penyakit.json"

# --- Spesifikasi model ---------------------------------------------------
# Urutan kelas WAJIB sama persis dengan hasil ImageFolder saat pelatihan.
# ImageFolder mengurutkan nama folder secara alfabetis, sehingga:
CLASS_NAMES = [
    "BrownSpot",
    "Healthy",
    "LeafBlast",
    "LeafBlight",
    "NotRiceLeaf",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224

# Statistik normalisasi ImageNet, sesuai bobot awal EfficientNet-B0.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# --- Ambang batas antarmuka ---------------------------------------------
# Prediksi di bawah ambang ini ditandai sebagai "kurang meyakinkan".
CONFIDENCE_THRESHOLD = 0.60

# Ukuran unggahan maksimum yang diterima (MB).
MAX_UPLOAD_MB = 10

# --- Metrik hasil eksperimen --------------------------------------------
# Diisi dari hasil eksperimen Bab IV. Ubah bila model dilatih ulang.
METRICS = {
    "internal": {
        "label": "Dataset RiceClass (validasi)",
        "jumlah_gambar": 1890,
        "accuracy": 0.9700,
        "precision": 0.97,
        "recall": 0.97,
        "f1": 0.97,
        "per_kelas": {
            "BrownSpot": 0.96,
            "Healthy": 1.00,
            "LeafBlast": 0.93,
            "LeafBlight": 0.97,
            "NotRiceLeaf": 0.99,
        },
    },
    "eksternal": {
        "label": "Cross-validation dataset",
        "jumlah_gambar": 1261,
        "accuracy": 0.8763,
        "precision": 0.88,
        "recall": 0.87,
        "f1": 0.87,
        "per_kelas": {
            "BrownSpot": 0.81,
            "Healthy": 0.95,
            "LeafBlast": 0.80,
            "LeafBlight": 0.78,
            "NotRiceLeaf": 1.00,
        },
    },
}

EFISIENSI = {
    "ukuran_model_mb": 16.4,
    "parameter_total": 4_218_851,
    "parameter_trainable": 1_340_695,
    "inference_ms": 8.63,
    "throughput_fps": 115.83,
}

ABLASI = {
    "tanpa_cbam": {
        "internal": 0.93,
        "eksternal": 0.6574,
        "f1_leafblight": 0.0000,
        "inference_ms": 7.78,
    },
    "dengan_cbam": {
        "internal": 0.97,
        "eksternal": 0.8763,
        "f1_leafblight": 0.7798,
        "inference_ms": 8.63,
    },
}
