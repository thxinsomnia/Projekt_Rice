"""Halaman hasil eksperimen: metrik, generalisasi, dan efisiensi."""

import streamlit as st

from config.settings import ABLASI, EFISIENSI, METRICS
from src.ui.components import (
    eyebrow,
    garis,
    judul,
    kartu_statistik,
    lead,
)
from src.ui.theme import WARNA


def _tabel_per_kelas() -> None:
    internal = METRICS["internal"]["per_kelas"]
    eksternal = METRICS["eksternal"]["per_kelas"]

    baris = ""
    for kelas in internal:
        nilai_int = internal[kelas]
        nilai_eks = eksternal[kelas]
        selisih = nilai_eks - nilai_int

        warna_selisih = WARNA["lesi"] if selisih < -0.05 else WARNA["tinta_redup"]
        tanda = "+" if selisih >= 0 else ""

        baris += f"""
        <tr>
            <td>{kelas}</td>
            <td class="rl-num">{nilai_int:.2f}</td>
            <td class="rl-num">{nilai_eks:.2f}</td>
            <td class="rl-num" style="color:{warna_selisih};">
                {tanda}{selisih:.2f}
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="rl-tabel">
            <thead>
                <tr>
                    <th>Kelas</th>
                    <th style="text-align:right;">F1 internal</th>
                    <th style="text-align:right;">F1 eksternal</th>
                    <th style="text-align:right;">Selisih</th>
                </tr>
            </thead>
            <tbody>{baris}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def _tabel_ablasi() -> None:
    tanpa = ABLASI["tanpa_cbam"]
    dengan = ABLASI["dengan_cbam"]

    barisan = [
        ("Akurasi dataset internal", f"{tanpa['internal']:.0%}", f"{dengan['internal']:.0%}"),
        ("Akurasi dataset eksternal", f"{tanpa['eksternal']:.2%}", f"{dengan['eksternal']:.2%}"),
        ("F1-score hawar daun bakteri", f"{tanpa['f1_leafblight']:.4f}", f"{dengan['f1_leafblight']:.4f}"),
        ("Waktu inferensi", f"{tanpa['inference_ms']:.2f} ms", f"{dengan['inference_ms']:.2f} ms"),
    ]

    isi = ""
    for label, nilai_tanpa, nilai_dengan in barisan:
        isi += f"""
        <tr>
            <td>{label}</td>
            <td class="rl-num">{nilai_tanpa}</td>
            <td class="rl-num" style="color:{WARNA["daun"]};font-weight:700;">
                {nilai_dengan}
            </td>
        </tr>
        """

    st.markdown(
        f"""
        <table class="rl-tabel">
            <thead>
                <tr>
                    <th>Metrik</th>
                    <th style="text-align:right;">Tanpa CBAM</th>
                    <th style="text-align:right;">Dengan CBAM</th>
                </tr>
            </thead>
            <tbody>{isi}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def tampilkan() -> None:
    eyebrow("Hasil eksperimen")
    st.markdown(
        """
        <h1 class="rl-display">Angka yang<br><em>bisa diperiksa</em></h1>
        """,
        unsafe_allow_html=True,
    )
    lead(
        "Seluruh metrik di bawah diperoleh dari pelatihan pada NVIDIA Tesla T4 "
        "dengan PyTorch. Model terbaik diambil dari epoch ke-15, dan pelatihan "
        "dihentikan otomatis pada epoch ke-27."
    )

    st.write("")

    kolom = st.columns(4, gap="medium")
    with kolom[0]:
        kartu_statistik(
            f"{METRICS['internal']['jumlah_gambar']:,}".replace(",", "."),
            "Gambar pada dataset validasi internal",
        )
    with kolom[1]:
        kartu_statistik(
            f"{METRICS['eksternal']['jumlah_gambar']:,}".replace(",", "."),
            "Gambar pada dataset eksternal",
        )
    with kolom[2]:
        kartu_statistik(
            f"{EFISIENSI['parameter_total'] / 1e6:.2f} jt",
            "Total parameter model",
        )
    with kolom[3]:
        kartu_statistik(
            f"{EFISIENSI['throughput_fps']:.0f} FPS",
            "Throughput pada perangkat pengujian",
        )

    garis()

    kiri, kanan = st.columns([1, 1], gap="large")

    with kiri:
        judul("Performa per kelas")
        st.markdown(
            f"""
            <p style="font-size:0.92rem;line-height:1.6;
                      color:{WARNA["tinta_redup"]};margin-bottom:1rem;">
                Selisih negatif yang besar menandakan kelas tersebut paling
                terpengaruh oleh perbedaan karakteristik gambar antar sumber
                data.
            </p>
            """,
            unsafe_allow_html=True,
        )
        _tabel_per_kelas()

    with kanan:
        judul("Pengaruh CBAM")
        st.markdown(
            f"""
            <p style="font-size:0.92rem;line-height:1.6;
                      color:{WARNA["tinta_redup"]};margin-bottom:1rem;">
                Kedua model dilatih dengan konfigurasi yang identik. Satu-satunya
                perbedaan adalah keberadaan modul attention.
            </p>
            """,
            unsafe_allow_html=True,
        )
        _tabel_ablasi()

    garis()

    judul("Batas yang perlu diketahui")

    kolom = st.columns(2, gap="large")

    with kolom[0]:
        st.markdown(
            f"""
            <div class="rl-kartu">
                <div class="rl-eyebrow" style="color:{WARNA["lesi"]};">
                    Pergeseran domain
                </div>
                <p style="font-size:0.92rem;line-height:1.62;
                          color:{WARNA["tinta_redup"]};margin:0;">
                    Akurasi turun sekitar 9 poin ketika model diuji pada gambar
                    dari sumber berbeda. Perbedaan pencahayaan, latar belakang,
                    dan kualitas kamera menjadi penyebab utamanya. Angka pada
                    dataset internal sebaiknya tidak dibaca sebagai perkiraan
                    performa di lahan sesungguhnya.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kolom[1]:
        st.markdown(
            f"""
            <div class="rl-kartu">
                <div class="rl-eyebrow" style="color:{WARNA["lesi"]};">
                    Kemiripan antar gejala
                </div>
                <p style="font-size:0.92rem;line-height:1.62;
                          color:{WARNA["tinta_redup"]};margin:0;">
                    Sebagian besar kesalahan terjadi antara bercak coklat, blas
                    daun, dan hawar daun bakteri. Analisis Grad-CAM menunjukkan
                    model tetap menyorot area lesi dengan benar, sehingga
                    kesalahan berasal dari kemiripan bentuk dan warna lesi,
                    bukan dari kegagalan model mendeteksi gejala penyakit.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
