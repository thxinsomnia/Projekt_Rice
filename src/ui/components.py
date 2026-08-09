"""Potongan tampilan yang dipakai berulang di beberapa halaman."""

import json
from functools import lru_cache

import streamlit as st

from config.settings import DISEASE_INFO_PATH
from src.ui.theme import TINGKAT_DAMPAK, WARNA


@lru_cache(maxsize=1)
def muat_info_penyakit() -> dict:
    """Membaca basis pengetahuan penyakit dari data/penyakit.json."""
    with open(DISEASE_INFO_PATH, encoding="utf-8") as berkas:
        return json.load(berkas)


def eyebrow(teks: str) -> None:
    st.markdown(f'<div class="rl-eyebrow">{teks}</div>', unsafe_allow_html=True)


def judul(teks: str) -> None:
    st.markdown(f'<h2 class="rl-judul">{teks}</h2>', unsafe_allow_html=True)


def lead(teks: str) -> None:
    st.markdown(f'<p class="rl-lead">{teks}</p>', unsafe_allow_html=True)


def garis() -> None:
    st.markdown('<hr class="rl-garis">', unsafe_allow_html=True)


def kartu_statistik(nilai: str, label: str) -> None:
    st.markdown(
        f"""
        <div class="rl-stat">
            <div class="rl-stat-nilai">{nilai}</div>
            <div class="rl-stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pil_dampak(tingkat: str) -> str:
    """Mengembalikan HTML label tingkat dampak."""
    teks, warna = TINGKAT_DAMPAK.get(tingkat, TINGKAT_DAMPAK["netral"])
    return f'<span class="rl-pil" style="color:{warna}">{teks}</span>'


def baris_probabilitas(nama: str, nilai: float, sorot: bool = False) -> str:
    """Satu baris batang probabilitas dalam bentuk HTML."""
    warna = WARNA["padi"] if sorot else WARNA["daun_muda"]
    persen = nilai * 100
    return f"""
    <div class="rl-baris">
        <div class="rl-baris-nama">{nama}</div>
        <div class="rl-baris-alur">
            <div class="rl-baris-isi"
                 style="width:{persen:.1f}%; background:{warna};"></div>
        </div>
        <div class="rl-baris-nilai">{persen:.1f}%</div>
    </div>
    """


def daftar(butir: list[str]) -> str:
    isi = "".join(f"<li>{item}</li>" for item in butir)
    return f'<ul class="rl-daftar">{isi}</ul>'


def kartu_penyakit(kunci: str, info: dict, ringkas: bool = False) -> None:
    """Kartu keterangan satu penyakit."""
    dampak = pil_dampak(info["dampak"])
    patogen = (
        f'<div style="font-size:0.82rem;font-style:italic;'
        f'color:{WARNA["tinta_redup"]};margin-bottom:0.5rem;">'
        f'{info["patogen"]}</div>'
        if info["patogen"] != "—"
        else ""
    )

    kehilangan = ""
    if info["kehilangan_hasil"] != "—":
        kehilangan = (
            f'<div style="font-family:JetBrains Mono,monospace;'
            f'font-size:0.78rem;color:{WARNA["lesi"]};margin-top:0.7rem;">'
            f'Potensi kehilangan hasil {info["kehilangan_hasil"]}</div>'
        )

    st.markdown(
        f"""
        <div class="rl-kartu">
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;gap:0.8rem;margin-bottom:0.5rem;">
                <div>
                    <div style="font-family:Fraunces,serif;font-size:1.22rem;
                                font-weight:600;color:{WARNA["sawah"]};">
                        {info["nama"]}
                    </div>
                    <div style="font-size:0.78rem;color:{WARNA["tinta_redup"]};">
                        {info["nama_en"]} &middot; {kunci}
                    </div>
                </div>
                {dampak}
            </div>
            {patogen}
            <p style="font-size:0.9rem;line-height:1.6;
                      color:{WARNA["tinta_redup"]};margin:0.6rem 0 0 0;">
                {info["ringkasan"]}
            </p>
            {kehilangan}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ringkas:
        return

    with st.expander("Gejala yang dikenali"):
        st.markdown(daftar(info["gejala"]), unsafe_allow_html=True)

    with st.expander("Cara penanggulangan"):
        st.markdown(daftar(info["penanggulangan"]), unsafe_allow_html=True)

def kartu_penyakit1(kunci: str, info: dict, ringkas: bool = False) -> None:
    """Kartu keterangan satu penyakit."""
    dampak = pil_dampak(info["dampak"])
    patogen = (
        f'<div style="font-size:0.82rem;font-style:italic;'
        f'color:{WARNA["tinta_redup"]};margin-bottom:0.5rem;">'
        f'{info["patogen"]}</div>'
        if info["patogen"] != "—"
        else ""
    )

    st.markdown(
        f"""
        <div class="rl-kartu">
            <div style="display:flex;justify-content:space-between;
                        align-items:flex-start;gap:0.8rem;margin-bottom:0.5rem;">
                <div>
                    <div style="font-family:Fraunces,serif;font-size:1.22rem;
                                font-weight:600;color:{WARNA["sawah"]};">
                        {info["nama"]}
                    </div>
                    <div style="font-size:0.78rem;color:{WARNA["tinta_redup"]};">
                        {info["nama_en"]} &middot; {kunci}
                    </div>
                </div>
                {dampak}
            </div>
            {patogen}
            <p style="font-size:0.9rem;line-height:1.6;
                      color:{WARNA["tinta_redup"]};margin:0.6rem 0 0 0;">
                {info["ringkasan"]}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ringkas:
        return

    with st.expander("Gejala yang dikenali"):
        st.markdown(daftar(info["gejala"]), unsafe_allow_html=True)

    with st.expander("Cara penanggulangan"):
        st.markdown(daftar(info["penanggulangan"]), unsafe_allow_html=True)

def langkah(nomor: int, judul_langkah: str, isi: str) -> None:
    st.markdown(
        f"""
        <div class="rl-langkah">
            <div class="rl-langkah-nomor">{nomor:02d}</div>
            <div>
                <div class="rl-langkah-judul">{judul_langkah}</div>
                <div class="rl-langkah-isi">{isi}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
