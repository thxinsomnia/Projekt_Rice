"""Halaman pembuka: pernyataan singkat dan angka-angka kunci."""

import streamlit as st

from config.settings import EFISIENSI, METRICS
from src.ui.components import (
    eyebrow,
    garis,
    judul,
    kartu_statistik,
    lead,
)
from src.ui.theme import WARNA


def tampilkan() -> None:
    eyebrow("Deteksi penyakit daun padi")
    st.markdown(
        """
        <h1 class="rl-display">
            Model ringan yang tahu<br><em>ke mana harus melihat</em>
        </h1>
        """,
        unsafe_allow_html=True,
    )
    lead(
        "EfficientNet-B0 yang dipadukan dengan Convolutional Block Attention "
        "Module untuk mengenali empat kondisi daun padi, ditambah satu kelas "
        "penolak untuk gambar di luar cakupan. Berukuran 16,4 MB dan berjalan "
        "di bawah 10 milidetik per gambar."
    )

    st.write("")

    kolom = st.columns(4, gap="medium")
    with kolom[0]:
        kartu_statistik(
            f"{METRICS['internal']['accuracy'] * 100:.0f}%",
            "Akurasi pada dataset validasi RiceClass",
        )
    with kolom[1]:
        kartu_statistik(
            f"{METRICS['eksternal']['accuracy'] * 100:.1f}%",
            "Akurasi pada dataset eksternal yang belum pernah dilatih",
        )
    with kolom[2]:
        kartu_statistik(
            f"{EFISIENSI['ukuran_model_mb']:.1f} MB",
            "Ukuran berkas model hasil pelatihan",
        )
    with kolom[3]:
        kartu_statistik(
            f"{EFISIENSI['inference_ms']:.2f} ms",
            "Waktu rata-rata memproses satu gambar",
        )

    garis()

    kiri, kanan = st.columns([1.25, 1], gap="large")

    with kiri:
        judul("Mengapa attention diperlukan")
        st.markdown(
            f"""
            <p class="rl-lead">
                Gejala bercak coklat, blas daun, dan hawar daun bakteri kerap
                terlihat mirip, terutama pada fase awal. Model tanpa mekanisme
                attention masih bisa mencapai akurasi tinggi di data
                pelatihannya sendiri, tetapi runtuh ketika dihadapkan pada
                gambar dari sumber yang berbeda.
            </p>
            <p class="rl-lead" style="margin-top:0.9rem;">
                Pada pengujian dataset eksternal, model tanpa CBAM gagal total
                mengenali seluruh gambar hawar daun bakteri. Setelah CBAM
                ditambahkan, kelas yang sama memperoleh F1-score 0,78 dengan
                tambahan biaya komputasi kurang dari satu milidetik.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with kanan:
        st.markdown(
            f"""
            <div class="rl-kartu-gelap">
                <div class="rl-eyebrow">Uji generalisasi</div>
                <table class="rl-tabel" style="color:{WARNA["kertas"]};">
                    <tr>
                        <td style="border-color:rgba(255,255,255,0.12);">
                            Tanpa CBAM
                        </td>
                        <td class="rl-num"
                            style="border-color:rgba(255,255,255,0.12);
                                   color:{WARNA["lesi"]};">
                            65,7%
                        </td>
                    </tr>
                    <tr>
                        <td style="border-color:rgba(255,255,255,0.12);">
                            Dengan CBAM
                        </td>
                        <td class="rl-num"
                            style="border-color:rgba(255,255,255,0.12);
                                   color:{WARNA["padi_terang"]};">
                            87,6%
                        </td>
                    </tr>
                </table>
                <div style="font-size:0.78rem;opacity:0.65;margin-top:0.9rem;
                            line-height:1.5;">
                    Akurasi pada 1.261 gambar dari sumber publik yang sama
                    sekali tidak dipakai selama pelatihan.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    garis()

    judul("Yang bisa dilakukan di sini")

    kolom = st.columns(3, gap="medium")

    isi = [
        (
            "Analisis gambar",
            "Unggah foto daun, dapatkan kelas penyakit beserta tingkat "
            "keyakinan dan sebaran probabilitas seluruh kelas.",
        ),
        (
            "Lacak keputusan model",
            "Peta Grad-CAM menunjukkan area yang menjadi dasar klasifikasi, "
            "diambil langsung dari keluaran modul CBAM.",
        ),
        (
            "Pakai bobot sendiri",
            "Model bawaan bisa diganti dengan berkas .pth hasil pelatihan "
            "lain lewat panel di samping kiri.",
        ),
    ]

    for kolom_ke, (nama, keterangan) in zip(kolom, isi):
        with kolom_ke:
            st.markdown(
                f"""
                <div class="rl-kartu">
                    <div style="font-family:Fraunces,serif;font-size:1.1rem;
                                font-weight:600;color:{WARNA["sawah"]};
                                margin-bottom:0.45rem;">
                        {nama}
                    </div>
                    <div style="font-size:0.9rem;line-height:1.6;
                                color:{WARNA["tinta_redup"]};">
                        {keterangan}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
