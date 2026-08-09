"""Halaman metodologi: urutan tahapan penelitian."""

import streamlit as st

from src.ui.components import eyebrow, garis, judul, langkah, lead
from src.ui.theme import WARNA

# Penomoran dipakai karena tahapan ini memang berurutan: keluaran satu
# tahap menjadi masukan tahap berikutnya.
TAHAPAN = [
    (
        "Akuisisi dataset",
        "Dataset RiceClass yang disesuaikan menjadi lima kelas, berjumlah "
        "8.660 gambar. Kelas NotRiceLeaf diambil dari sebagian ImageNet untuk "
        "melatih model menolak masukan di luar cakupan.",
    ),
    (
        "Pembagian data",
        "Hold-out validation dengan rasio 80 berbanding 20. Dataset "
        "cross-validation dari empat sumber publik lain disimpan terpisah dan "
        "tidak pernah disentuh selama pelatihan.",
    ),
    (
        "Prapemrosesan",
        "Data pelatihan melewati random resized crop, flip horizontal dan "
        "vertikal, rotasi, color jitter, serta affine. Data validasi hanya "
        "diubah ukurannya dan dinormalisasi agar mewakili kondisi sebenarnya.",
    ),
    (
        "Perancangan model",
        "EfficientNet-B0 sebagai backbone, dengan CBAM disisipkan setelah "
        "proses ekstraksi fitur pada 1.280 channel, sebelum lapisan "
        "klasifikasi.",
    ),
    (
        "Transfer learning",
        "Bobot awal ImageNet dengan partial fine-tuning. Sebagian besar "
        "backbone dibekukan berikut statistik Batch Normalization-nya; hanya "
        "dua blok terakhir, CBAM, dan classifier yang dilatih.",
    ),
    (
        "Pelatihan",
        "Adam dengan learning rate 0,0001 dan weight decay 0,0001, "
        "CrossEntropyLoss, penjadwalan ReduceLROnPlateau, serta early stopping "
        "dengan kesabaran tujuh epoch.",
    ),
    (
        "Evaluasi",
        "Accuracy, precision, recall, F1-score, dan confusion matrix pada dua "
        "dataset, ditambah pengukuran ukuran model, jumlah parameter, waktu "
        "inferensi, dan throughput.",
    ),
    (
        "Interpretasi",
        "Grad-CAM diarahkan ke keluaran modul CBAM untuk memastikan keputusan "
        "klasifikasi bersandar pada area lesi, bukan pada latar belakang.",
    ),
]


def tampilkan() -> None:
    eyebrow("Pendekatan teknis")
    st.markdown(
        """
        <h1 class="rl-display">Delapan tahap,<br><em>berurutan</em></h1>
        """,
        unsafe_allow_html=True,
    )
    lead(
        "Setiap tahap menghasilkan keluaran yang menjadi masukan tahap "
        "berikutnya. Konfigurasi lengkapnya tercatat di berkas notebook "
        "pelatihan."
    )

    st.write("")

    for nomor, (nama, keterangan) in enumerate(TAHAPAN, start=1):
        langkah(nomor, nama, keterangan)

    garis()

    judul("Perangkat yang dipakai")

    kolom = st.columns(2, gap="large")

    spesifikasi = [
        ("Lingkungan", "Google Colab"),
        ("GPU", "NVIDIA Tesla T4"),
        ("Framework", "PyTorch"),
        ("Bahasa", "Python"),
        ("Antarmuka", "Streamlit"),
        ("Ukuran masukan", "224 × 224 piksel"),
    ]

    tengah = len(spesifikasi) // 2
    for kolom_ke, bagian in zip(kolom, [spesifikasi[:tengah], spesifikasi[tengah:]]):
        with kolom_ke:
            baris = "".join(
                f"""<tr>
                    <td style="color:{WARNA["tinta_redup"]};">{label}</td>
                    <td style="text-align:right;font-weight:500;">{nilai}</td>
                </tr>"""
                for label, nilai in bagian
            )
            st.markdown(
                f'<table class="rl-tabel">{baris}</table>',
                unsafe_allow_html=True,
            )
