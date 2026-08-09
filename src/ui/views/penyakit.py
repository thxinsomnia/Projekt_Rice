"""Halaman katalog: seluruh kelas yang dikenali model."""

import streamlit as st

from src.ui.components import (
    eyebrow,
    garis,
    kartu_penyakit,
    kartu_penyakit1,
    lead,
    muat_info_penyakit,
)

# Urutan tampilan disusun dari yang paling merugikan, bukan alfabetis,
# karena halaman ini dibaca sebagai rujukan lapangan.
URUTAN_TAMPILAN = [
    "LeafBlight",
    "LeafBlast",
    "BrownSpot",
   
]

URUTANTAMPILAN = ["Healthy", "NotRiceLeaf"]  

def tampilkan() -> None:
    eyebrow("Cakupan klasifikasi")
    st.markdown(
        """
        <h1 class="rl-display">Lima kondisi<br><em>yang dikenali</em></h1>
        """,
        unsafe_allow_html=True,
    )
    lead(
        "Empat kelas berkaitan dengan kondisi daun padi, ditambah satu kelas "
        "penolak agar model tidak memaksakan label penyakit pada gambar yang "
        "bukan daun padi. Urutan di bawah disusun dari yang paling merugikan "
        "hasil panen."
    )

    garis()

    info_semua = muat_info_penyakit()

    for kunci in URUTAN_TAMPILAN:
        info = info_semua[kunci]
        kartu_penyakit(kunci, info)
        st.write("")

    for kunci in URUTANTAMPILAN:
        info = info_semua[kunci]
        kartu_penyakit1(kunci, info)
        st.write("")

    st.caption(
        "Anjuran penanggulangan bersifat umum. Dosis dan bahan aktif "
        "pestisida sebaiknya disesuaikan dengan rekomendasi dinas pertanian "
        "setempat serta varietas yang ditanam."
    )

