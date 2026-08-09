"""Token desain dan gaya kustom.

Palet diambil dari siklus tanam padi: tanah sawah yang basah, hijau daun,
bulir yang menguning, dan warna lesi. Kuning padi dipakai sebagai aksen
utama, bukan hijau, agar penanda penting tidak melebur dengan latar.
"""

import streamlit as st

WARNA = {
    "sawah": "#14261C",  # tanah sawah basah, hampir hitam
    "daun": "#2F6B3F",  # hijau daun padi
    "daun_muda": "#5B9668",
    "padi": "#C8961E",  # bulir menguning, aksen utama
    "padi_terang": "#E8B84B",
    "lesi": "#A8442A",  # coklat kemerahan lesi, penanda bahaya
    "kertas": "#F4F6F0",  # latar, putih dengan semburat hijau
    "kertas_gelap": "#E7EBE1",
    "tinta": "#1B241D",
    "tinta_redup": "#5C6A5E",
    "garis": "#D3DACB",
}

TINGKAT_DAMPAK = {
    "tinggi": ("Dampak tinggi", WARNA["lesi"]),
    "sedang": ("Dampak sedang", WARNA["padi"]),
    "aman": ("Kondisi baik", WARNA["daun"]),
    "netral": ("Di luar cakupan", WARNA["tinta_redup"]),
}


def terapkan_tema() -> None:
    """Menyuntikkan font dan gaya kustom ke halaman."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

        :root {{
            --sawah: {WARNA["sawah"]};
            --daun: {WARNA["daun"]};
            --daun-muda: {WARNA["daun_muda"]};
            --padi: {WARNA["padi"]};
            --padi-terang: {WARNA["padi_terang"]};
            --lesi: {WARNA["lesi"]};
            --kertas: {WARNA["kertas"]};
            --kertas-gelap: {WARNA["kertas_gelap"]};
            --tinta: {WARNA["tinta"]};
            --tinta-redup: {WARNA["tinta_redup"]};
            --garis: {WARNA["garis"]};
        }}

        .stApp {{
            background: var(--kertas);
        }}

        html, body, [class*="css"] {{
            font-family: 'Inter', system-ui, sans-serif;
            color: var(--tinta);
        }}

        .block-container {{
            padding-top: 2.5rem;
            padding-bottom: 4rem;
            max-width: 1140px;
        }}

        /* --- Tipografi ------------------------------------------------ */
        .rl-display {{
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 700;
            font-size: clamp(2.4rem, 5vw, 3.9rem);
            line-height: 1.02;
            letter-spacing: -0.022em;
            color: var(--sawah);
            margin: 0 0 1rem 0;
        }}

        .rl-display em {{
            font-style: italic;
            font-weight: 400;
            color: var(--daun);
        }}

        .rl-judul {{
            font-family: 'Fraunces', Georgia, serif;
            font-weight: 600;
            font-size: clamp(1.6rem, 3vw, 2.3rem);
            line-height: 1.14;
            letter-spacing: -0.015em;
            color: var(--sawah);
            margin: 0 0 0.6rem 0;
        }}

        .rl-eyebrow {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--daun);
            margin-bottom: 0.7rem;
        }}

        .rl-lead {{
            font-size: 1.06rem;
            line-height: 1.62;
            color: var(--tinta-redup);
            max-width: 60ch;
        }}

        .rl-angka {{
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            font-variant-numeric: tabular-nums;
        }}

        /* --- Kartu ----------------------------------------------------- */
        .rl-kartu {{
            background: #FFFFFF;
            border: 1px solid var(--garis);
            border-radius: 4px;
            padding: 1.4rem 1.5rem;
            height: 100%;
        }}

        .rl-kartu-gelap {{
            background: var(--sawah);
            border-radius: 4px;
            padding: 1.5rem 1.6rem;
            color: var(--kertas);
        }}

        .rl-kartu-gelap .rl-eyebrow {{ color: var(--padi-terang); }}

        /* Garis aksen tipis di sisi kiri kartu statistik */
        .rl-stat {{
            background: #FFFFFF;
            border: 1px solid var(--garis);
            border-left: 3px solid var(--padi);
            border-radius: 3px;
            padding: 1.1rem 1.25rem;
        }}

        .rl-stat-nilai {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--sawah);
            font-variant-numeric: tabular-nums;
            line-height: 1.1;
        }}

        .rl-stat-label {{
            font-size: 0.78rem;
            color: var(--tinta-redup);
            margin-top: 0.3rem;
            line-height: 1.4;
        }}

        /* --- Label kecil ----------------------------------------------- */
        .rl-pil {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            padding: 0.22rem 0.6rem;
            border-radius: 2px;
            border: 1px solid currentColor;
        }}

        /* --- Baris probabilitas ---------------------------------------- */
        .rl-baris {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.55rem;
        }}

        .rl-baris-nama {{
            flex: 0 0 9.5rem;
            font-size: 0.86rem;
            color: var(--tinta);
        }}

        .rl-baris-alur {{
            flex: 1;
            height: 7px;
            background: var(--kertas-gelap);
            border-radius: 1px;
            overflow: hidden;
        }}

        .rl-baris-isi {{
            height: 100%;
            border-radius: 1px;
        }}

        .rl-baris-nilai {{
            flex: 0 0 3.6rem;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-variant-numeric: tabular-nums;
            color: var(--tinta-redup);
        }}

        /* --- Daftar ----------------------------------------------------- */
        .rl-daftar {{
            margin: 0;
            padding-left: 1.05rem;
            font-size: 0.92rem;
            line-height: 1.68;
            color: var(--tinta);
        }}

        .rl-daftar li {{ margin-bottom: 0.38rem; }}
        .rl-daftar::marker {{ color: var(--padi); }}

        /* --- Pemisah ---------------------------------------------------- */
        .rl-garis {{
            border: none;
            border-top: 1px solid var(--garis);
            margin: 2.4rem 0 2rem 0;
        }}

        /* --- Langkah bernomor (urutan pipeline benar-benar berurutan) ---- */
        .rl-langkah {{
            display: flex;
            gap: 1.1rem;
            padding: 1.1rem 0;
            border-top: 1px solid var(--garis);
        }}

        .rl-langkah-nomor {{
            flex: 0 0 2.6rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            color: var(--padi);
            padding-top: 0.2rem;
        }}

        .rl-langkah-judul {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.12rem;
            font-weight: 600;
            color: var(--sawah);
            margin-bottom: 0.3rem;
        }}

        .rl-langkah-isi {{
            font-size: 0.92rem;
            line-height: 1.62;
            color: var(--tinta-redup);
        }}

        /* --- Panel samping ---------------------------------------------- */
        section[data-testid="stSidebar"] {{
            background: var(--sawah);
            border-right: none;
        }}

        section[data-testid="stSidebar"] * {{ color: var(--kertas); }}

        section[data-testid="stSidebar"] .rl-sidebar-merek {{
            font-family: 'Fraunces', Georgia, serif;
            font-size: 1.28rem;
            font-weight: 600;
            line-height: 1.2;
            color: #FFFFFF;
            margin-bottom: 0.15rem;
        }}

        section[data-testid="stSidebar"] .rl-sidebar-sub {{
            font-size: 0.76rem;
            color: var(--daun-muda);
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 1.4rem;
        }}

        /* --- Tombol ------------------------------------------------------ */
        .stButton > button {{
            background: var(--sawah);
            color: var(--kertas);
            border: 1px solid var(--sawah);
            border-radius: 3px;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0.55rem 1.4rem;
            transition: background 0.15s ease, color 0.15s ease;
        }}

        .stButton > button:hover {{
            background: var(--daun);
            border-color: var(--daun);
            color: #FFFFFF;
        }}

        .stButton > button:focus-visible {{
            outline: 2px solid var(--padi);
            outline-offset: 2px;
        }}

        /* --- Tab --------------------------------------------------------- */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 1.6rem;
            border-bottom: 1px solid var(--garis);
        }}

        .stTabs [data-baseweb="tab"] {{
            font-size: 0.88rem;
            font-weight: 600;
            color: var(--tinta-redup);
            padding: 0.5rem 0;
        }}

        .stTabs [aria-selected="true"] {{ color: var(--sawah); }}

        /* --- Tabel ------------------------------------------------------- */
        .rl-tabel {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        .rl-tabel th {{
            text-align: left;
            font-size: 0.72rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--tinta-redup);
            font-weight: 600;
            padding: 0.5rem 0.7rem;
            border-bottom: 1px solid var(--garis);
        }}

        .rl-tabel td {{
            padding: 0.6rem 0.7rem;
            border-bottom: 1px solid var(--kertas-gelap);
        }}

        .rl-tabel td.rl-num {{
            font-family: 'JetBrains Mono', monospace;
            font-variant-numeric: tabular-nums;
            text-align: right;
        }}

        /* --- Aksesibilitas ------------------------------------------------ */
        @media (prefers-reduced-motion: reduce) {{
            * {{ transition: none !important; animation: none !important; }}
        }}

        #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
