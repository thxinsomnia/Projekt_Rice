"""Aplikasi deteksi penyakit daun padi berbasis EfficientNet-B0 + CBAM.

Jalankan dengan:
    streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Pastikan paket lokal dapat diimpor ketika dijalankan dari mana saja.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import DEFAULT_MODEL_PATH  # noqa: E402
from src.model.loader import ambil_model, model_bawaan_tersedia  # noqa: E402
from src.ui.theme import WARNA, terapkan_tema  # noqa: E402
from src.ui.views import (  # noqa: E402
    beranda,
    deteksi,
    metodologi,
    penyakit,
    performa,
)

HALAMAN = {
    "Beranda": beranda,
    "Deteksi": deteksi,
    "Penyakit": penyakit,
    "Performa": performa,
    "Metodologi": metodologi,
}


def siapkan_halaman() -> None:
    st.set_page_config(
        page_title="Deteksi Penyakit Daun Padi",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    terapkan_tema()


def panel_samping() -> tuple[str, object]:
    """Merender panel kiri dan mengembalikan (halaman terpilih, model)."""
    with st.sidebar:
        st.markdown(
            """
            <div class="rl-sidebar-merek">Deteksi Daun Padi</div>
            <div class="rl-sidebar-sub">EfficientNet-B0 + CBAM</div>
            """,
            unsafe_allow_html=True,
        )

        halaman = st.radio(
            "Halaman",
            list(HALAMAN.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Sumber bobot model**")

        pilihan = st.radio(
            "Sumber bobot model",
            ["Model bawaan", "Unggah model sendiri"],
            label_visibility="collapsed",
        )

        berkas_model = None

        if pilihan == "Unggah model sendiri":
            berkas_model = st.file_uploader(
                "Berkas bobot",
                type=["pth", "pt"],
                help=(
                    "State dict PyTorch dari arsitektur EfficientNet-B0 + CBAM "
                    "dengan 5 kelas."
                ),
            )
            if berkas_model is None:
                st.caption(
                    "Belum ada berkas. Aplikasi sementara memakai model bawaan."
                )
        else:
            if model_bawaan_tersedia():
                st.caption(f"Memakai `{DEFAULT_MODEL_PATH.name}`")
            else:
                st.caption(
                    f"`{DEFAULT_MODEL_PATH.name}` belum ada di folder models/"
                )

    return halaman, berkas_model


def kartu_status_model(dimuat) -> None:
    """Ringkasan model aktif di bagian bawah panel samping."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size:0.72rem;letter-spacing:0.12em;
                        text-transform:uppercase;color:{WARNA["daun_muda"]};
                        margin-bottom:0.5rem;">
                Model aktif
            </div>
            <div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;
                        word-break:break-all;margin-bottom:0.6rem;">
                {dimuat.sumber}
            </div>
            <div style="font-size:0.75rem;opacity:0.65;line-height:1.6;">
                {dimuat.jumlah_parameter:,} parameter<br>
                {dimuat.ukuran_mb:.1f} MB &middot; {dimuat.device.type.upper()}
            </div>
            """.replace(
                ",", "."
            ),
            unsafe_allow_html=True,
        )


def main() -> None:
    siapkan_halaman()

    halaman, berkas_model = panel_samping()

    dimuat, kesalahan = ambil_model(berkas_model)

    if dimuat is not None:
        kartu_status_model(dimuat)

    modul = HALAMAN[halaman]

    # Hanya halaman Deteksi yang membutuhkan model termuat.
    if halaman == "Deteksi":
        if dimuat is None:
            st.error(kesalahan, icon="🚫")
            st.markdown(
                f"""
                <div class="rl-kartu" style="margin-top:1rem;">
                    <div class="rl-eyebrow">Cara melengkapi</div>
                    <ol class="rl-daftar">
                        <li>Salin berkas <code>{DEFAULT_MODEL_PATH.name}</code>
                            hasil pelatihan ke folder <code>models/</code>.</li>
                        <li>Atau pilih <em>Unggah model sendiri</em> di panel
                            kiri, lalu pilih berkas <code>.pth</code>.</li>
                        <li>Muat ulang halaman setelah berkas tersedia.</li>
                    </ol>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return
        modul.tampilkan(dimuat)
    else:
        if kesalahan and halaman == "Beranda":
            st.warning(kesalahan, icon="⚠️")
        modul.tampilkan()


if __name__ == "__main__":
    main()
