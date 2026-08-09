"""Halaman utama: menganalisis satu gambar daun."""

import streamlit as st

from config.settings import CONFIDENCE_THRESHOLD, MAX_UPLOAD_MB
from src.inference.gradcam import tumpuk_pada_gambar
from src.inference.predictor import prediksi
from src.inference.preprocess import buka_gambar
from src.model.loader import LoadedModel
from src.ui.components import (
    baris_probabilitas,
    daftar,
    eyebrow,
    garis,
    judul,
    lead,
    muat_info_penyakit,
    pil_dampak,
)
from src.ui.theme import WARNA


def _panel_hasil(hasil, info: dict) -> None:
    """Kartu ringkas berisi kelas terpilih dan tingkat keyakinannya."""
    meyakinkan = hasil.confidence >= CONFIDENCE_THRESHOLD
    warna_nilai = WARNA["padi_terang"] if meyakinkan else WARNA["lesi"]

    st.markdown(
        f"""
        <div class="rl-kartu-gelap">
            <div class="rl-eyebrow">Hasil deteksi</div>
            <div style="font-family:Fraunces,serif;font-size:2rem;
                        font-weight:700;line-height:1.1;margin-bottom:0.15rem;">
                {info["nama"]}
            </div>
            <div style="font-size:0.82rem;opacity:0.7;margin-bottom:1.1rem;">
                {info["nama_en"]} &middot; {hasil.kelas}
            </div>
            <div style="display:flex;align-items:baseline;gap:0.5rem;">
                <span class="rl-angka" style="font-size:2.6rem;
                      color:{warna_nilai};">
                    {hasil.confidence * 100:.1f}%
                </span>
                <span style="font-size:0.82rem;opacity:0.7;">
                    tingkat keyakinan
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not meyakinkan:
        st.warning(
            f"Keyakinan model di bawah {CONFIDENCE_THRESHOLD * 100:.0f}%. "
            f"Kemungkinan lain yang mendekati: **{hasil.runner_up[0]}** "
            f"({hasil.runner_up[1] * 100:.1f}%). Ambil ulang foto dengan "
            "pencahayaan merata dan fokus pada area bergejala.",
            icon="⚠️",
        )


def _panel_probabilitas(hasil) -> None:
    baris = "".join(
        baris_probabilitas(nama, nilai, sorot=(nama == hasil.kelas))
        for nama, nilai in hasil.peringkat
    )
    st.markdown(
        f"""
        <div class="rl-kartu">
            <div class="rl-eyebrow">Sebaran probabilitas</div>
            {baris} <div style="font-size:0.78rem;color:{WARNA["tinta_redup"]};margin-top:0.8rem;padding-top:0.7rem;border-top:1px solid {WARNA["garis"]};">
                Selisih dengan kandidat kedua
                <span class="rl-angka">
                    {hasil.selisih_dengan_runner_up * 100:.1f}
                </span> poin &middot;
                waktu inferensi
                <span class="rl-angka">{hasil.waktu_ms:.1f}</span> ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _panel_gradcam(hasil, gambar) -> None:
    """Visualisasi Grad-CAM dengan kendali opasitas."""
    judul("Bagian mana yang dilihat model")
    lead(
        "Grad-CAM menandai area gambar yang paling memengaruhi keputusan. "
        "Merah berarti pengaruh besar, biru pengaruh kecil. Geser kendali "
        "di bawah untuk membandingkan dengan gambar aslinya."
    )

    opasitas = st.slider(
        "Kepekatan heatmap",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Geser ke kiri untuk melihat gambar asli, "
        "ke kanan untuk melihat peta perhatian model.",
    )

    tumpukan = tumpuk_pada_gambar(
        hasil.gambar_dasar,
        hasil.heatmap,
        opasitas=opasitas,
    )

    kolom = st.columns(2, gap="medium")
    with kolom[0]:
        st.image(gambar, caption="Gambar yang dikirim", use_container_width=True)
    with kolom[1]:
        st.image(
            tumpukan,
            caption=f"Grad-CAM pada kelas {hasil.kelas}",
            use_container_width=True,
        )

    st.caption(
        "Peta perhatian diambil dari keluaran modul CBAM, sehingga area yang "
        "ditampilkan sudah mencerminkan hasil penerapan attention."
    )


def _panel_penanganan(info: dict, kunci: str) -> None:
    garis()

    judul("Gejala dan penanganan")

    kolom = st.columns(2, gap="large")

    with kolom[0]:
        st.markdown(
            f"""
            <div class="rl-eyebrow">Ciri yang dikenali</div>
            {daftar(info["gejala"])}
            """,
            unsafe_allow_html=True,
        )

    with kolom[1]:
        st.markdown(
            f"""
            <div class="rl-eyebrow">Langkah penanggulangan</div>
            {daftar(info["penanggulangan"])}
            """,
            unsafe_allow_html=True,
        )

    keterangan = []
    if info["patogen"] != "—":
        keterangan.append(f"Penyebab: *{info['patogen']}* ({info['jenis']})")
    if info.get("kehilangan_hasil", "—") != "—":
        keterangan.append(
            f"Potensi kehilangan hasil {info['kehilangan_hasil']}"
        )

    if keterangan:
        st.markdown(
            f'<div style="margin-top:1rem;">{pil_dampak(info["dampak"])}</div>',
            unsafe_allow_html=True,
        )
        st.caption(" · ".join(keterangan))

    with st.expander("Catatan performa model pada kelas ini"):
        st.write(info["catatan_model"])

    st.info(
        "Hasil ini bersifat bantuan awal, bukan pengganti pemeriksaan "
        "penyuluh pertanian. Untuk keputusan penanganan di lahan luas, "
        "konfirmasi dulu dengan petugas setempat.",
        icon="ℹ️",
    )


def _keadaan_kosong() -> None:
    st.markdown(
        f"""
        <div class="rl-kartu" style="border-style:dashed;text-align:center;
                                     padding:3rem 1.5rem;">
            <div style="font-family:Fraunces,serif;font-size:1.15rem;
                        color:{WARNA["sawah"]};margin-bottom:0.4rem;">
                Belum ada gambar untuk dianalisis
            </div>
            <div style="font-size:0.9rem;color:{WARNA["tinta_redup"]};
                        max-width:34ch;margin:0 auto;line-height:1.6;">
                Unggah satu foto helai daun padi. Hasil deteksi, sebaran
                probabilitas, dan peta perhatian model akan muncul di sini.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tampilkan(dimuat: LoadedModel) -> None:
    """Merender seluruh halaman deteksi."""
    eyebrow("Demonstrasi model")
    st.markdown(
        '<h1 class="rl-display">Periksa satu helai<br><em>daun padi</em></h1>',
        unsafe_allow_html=True,
    )
    lead(
        "Unggah foto daun, lalu jalankan analisis. Model mengembalikan kelas "
        "penyakit, tingkat keyakinan, dan peta area yang menjadi dasar "
        "keputusannya."
    )

    st.write("")

    berkas = st.file_uploader(
        "Pilih gambar daun padi",
        type=["jpg", "jpeg", "png", "webp"],
        help=f"Format JPG, PNG, atau WebP. Ukuran maksimum {MAX_UPLOAD_MB} MB.",
    )

    if berkas is None:
        st.write("")
        _keadaan_kosong()
        return

    ukuran_mb = berkas.size / (1024**2)
    if ukuran_mb > MAX_UPLOAD_MB:
        st.error(
            f"Ukuran berkas {ukuran_mb:.1f} MB melewati batas "
            f"{MAX_UPLOAD_MB} MB. Perkecil resolusi gambar lalu unggah lagi."
        )
        return

    try:
        gambar = buka_gambar(berkas)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gambar tidak dapat dibaca: {exc}")
        return

    pratinjau, kendali = st.columns([1, 1.4], gap="large")

    with pratinjau:
        st.image(gambar, use_container_width=True)
        st.caption(
            f"{berkas.name} &middot; {gambar.width} × {gambar.height} piksel"
        )

    with kendali:
        st.markdown(
            f"""
            <div style="font-size:0.9rem;color:{WARNA["tinta_redup"]};
                        line-height:1.6;margin-bottom:1rem;">
                Gambar akan diubah ke ukuran 224 × 224 piksel dan dinormalisasi
                mengikuti statistik ImageNet, sama persis dengan prapemrosesan
                pada tahap validasi.
            </div>
            """,
            unsafe_allow_html=True,
        )
        jalankan = st.button("Analisis gambar", type="primary")

    id_gambar = f"{berkas.name}-{berkas.size}"

    if jalankan:
        with st.spinner("Menjalankan model…"):
            try:
                st.session_state["hasil_prediksi"] = prediksi(
                    dimuat, gambar, hitung_gradcam=True
                )
                st.session_state["id_gambar_diproses"] = id_gambar
            except Exception as exc:  # noqa: BLE001
                st.error(f"Analisis gagal dijalankan: {exc}")
                return

    hasil = st.session_state.get("hasil_prediksi")
    id_diproses = st.session_state.get("id_gambar_diproses")

    if hasil is None or id_diproses != id_gambar:
        return
    
    info_semua = muat_info_penyakit()
    info = info_semua[hasil.kelas]

    garis()

    kiri, kanan = st.columns([1, 1], gap="large")
    with kiri:
        _panel_hasil(hasil, info)
    with kanan:
        _panel_probabilitas(hasil)

    st.write("")
    garis()

    if hasil.heatmap is not None:
        _panel_gradcam(hasil, gambar)

    _panel_penanganan(info, hasil.kelas)
