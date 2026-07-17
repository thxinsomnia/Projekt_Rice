"""
app.py
------
Streamlit front-end for the Rice Leaf Disease Classifier
(EfficientNet-B0 + CBAM) with Grad-CAM explainability.

Run with:
    streamlit run app.py
"""

import os

import numpy as np
import streamlit as st
import torch
from PIL import Image

from config import (
    CLASS_NAMES,
    NUM_CLASSES,
    IMAGE_SIZE,
    DEFAULT_CHECKPOINT_PATH,
    WEIGHTS_DIR,
    CBAM_CHANNELS,
    CBAM_REDUCTION,
    CBAM_SPATIAL_KERNEL,
    DISEASE_INFO,
)
from models import build_model
from utils import preprocess_image, predict_image, get_gradcam, generate_gradcam_overlay


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSS_PATH = os.path.join(BASE_DIR, "assets", "style.css")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deteksi Penyakit Daun Padi | AI",
    page_icon="🌾",
    layout="wide",
)


def load_css(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css(CSS_PATH)


# ---------------------------------------------------------------------------
# Cached resources: model + Grad-CAM object
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model...")
def load_model(checkpoint_path, device_str):
    device = torch.device(device_str)
    model = build_model(
        num_classes=NUM_CLASSES,
        checkpoint_path=checkpoint_path,
        device=device,
        cbam_channels=CBAM_CHANNELS,
        cbam_reduction=CBAM_REDUCTION,
        cbam_spatial_kernel=CBAM_SPATIAL_KERNEL,
    )
    return model


@st.cache_resource(show_spinner=False)
def load_cam(_model):
    # underscore prefix tells Streamlit not to hash this arg
    return get_gradcam(_model)


# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Pengaturan")

device_option = st.sidebar.radio(
    "Perangkat",
    options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model")

uploaded_ckpt = st.sidebar.file_uploader(
    "Unggah checkpoint .pth (opsional)",
    type=["pth", "pt"],
    help="Jika belum ada file di folder weights/, unggah bobot model terlatih Anda di sini.",
)

checkpoint_path = DEFAULT_CHECKPOINT_PATH

if uploaded_ckpt is not None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    saved_ckpt_path = os.path.join(WEIGHTS_DIR, uploaded_ckpt.name)
    with open(saved_ckpt_path, "wb") as f:
        f.write(uploaded_ckpt.getbuffer())
    checkpoint_path = saved_ckpt_path
    st.sidebar.success(f"Menggunakan checkpoint: {uploaded_ckpt.name}")
elif os.path.exists(DEFAULT_CHECKPOINT_PATH):
    st.sidebar.info(f"Checkpoint aktif:\n`{os.path.basename(DEFAULT_CHECKPOINT_PATH)}`")
else:
    st.sidebar.warning(
        "Belum ada checkpoint di folder weights/ maupun yang diunggah. "
        "Model akan berjalan dengan bobot ImageNet saja "
        "(prediksi belum akurat sampai Anda menyediakan checkpoint terlatih)."
    )
    checkpoint_path = None

st.sidebar.markdown("---")
st.sidebar.subheader("Kelas Deteksi")
for cname in CLASS_NAMES:
    info = DISEASE_INFO.get(cname, {})
    st.sidebar.write(f"{info.get('icon', '•')} {info.get('label', cname)}")


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-banner">
        <span class="hero-badge">🌿 Didukung Deep Learning</span>
        <div class="hero-title">Deteksi Penyakit Daun Padi dengan AI</div>
        <p class="hero-subtitle">
            Unggah foto daun padi Anda dan biarkan model EfficientNet-B0 + CBAM
            mendeteksi jenis penyakitnya secara instan, lengkap dengan visualisasi
            Grad-CAM yang menunjukkan area yang menjadi fokus model.
        </p>
        <div class="hero-stats">
            <div>
                <div class="hero-stat-value">{len(CLASS_NAMES)}</div>
                <div class="hero-stat-label">Kelas Terdeteksi</div>
            </div>
            <div>
                <div class="hero-stat-value">224×224</div>
                <div class="hero-stat-label">Resolusi Input</div>
            </div>
            <div>
                <div class="hero-stat-value">Grad-CAM</div>
                <div class="hero-stat-label">Visualisasi Model</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Disease info cards
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">🔎 Jenis yang Dapat Dideteksi</div>', unsafe_allow_html=True)

info_cols = st.columns(len(CLASS_NAMES))
for col, cname in zip(info_cols, CLASS_NAMES):
    info = DISEASE_INFO.get(cname, {"icon": "🌱", "label": cname, "desc": ""})
    with col:
        st.markdown(
            f"""
            <div class="disease-card">
                <div class="disease-icon">{info['icon']}</div>
                <div class="disease-name">{info['label']}</div>
                <div class="disease-desc">{info['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")


# ---------------------------------------------------------------------------
# Upload & analysis section
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">📤 Unggah &amp; Analisis Gambar</div>', unsafe_allow_html=True)

st.markdown('<div class="app-card">', unsafe_allow_html=True)
uploaded_image = st.file_uploader(
    "Pilih gambar daun padi (JPG, PNG, atau WEBP)",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)
st.markdown('</div>', unsafe_allow_html=True)

run_button_placeholder = st.empty()

if uploaded_image is not None:
    pil_image = Image.open(uploaded_image).convert("RGB")

    col_input, col_result = st.columns(2)

    with col_input:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown("**Gambar Masukan**")
        st.image(pil_image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_clicked = run_button_placeholder.button("🔍 Analisis Gambar", type="primary")

    if analyze_clicked:
        with st.spinner("Memuat model dan menjalankan prediksi..."):
            model = load_model(checkpoint_path, device_option)
            cam = load_cam(model)

            input_tensor = preprocess_image(pil_image, image_size=IMAGE_SIZE)

            predicted_name, confidence, probabilities = predict_image(
                model, input_tensor, CLASS_NAMES, device=device_option
            )

            overlay, original_image = generate_gradcam_overlay(
                cam, input_tensor.to(device_option)
            )

        predicted_info = DISEASE_INFO.get(
            predicted_name, {"icon": "🌱", "label": predicted_name, "desc": ""}
        )

        with col_result:
            st.markdown('<div class="app-card">', unsafe_allow_html=True)
            st.markdown("**Hasil Prediksi**")

            pill_class = (
                "result-pill-healthy"
                if predicted_name.lower() == "healthy"
                else "result-pill-disease"
            )
            st.markdown(
                f"""
                <div class="result-pill {pill_class}">
                    {predicted_info['icon']} {predicted_info['label']} — {confidence:.2f}%
                </div>
                """,
                unsafe_allow_html=True,
            )

            if predicted_info.get("desc"):
                st.caption(predicted_info["desc"])

            st.write("**Probabilitas tiap kelas:**")
            for class_name, prob in sorted(
                probabilities.items(), key=lambda kv: kv[1], reverse=True
            ):
                label = DISEASE_INFO.get(class_name, {}).get("label", class_name)
                st.progress(min(int(prob), 100), text=f"{label}: {prob:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧠 Visualisasi Grad-CAM</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        gradcam_col1, gradcam_col2 = st.columns(2)

        with gradcam_col1:
            st.image(
                (original_image * 255).astype(np.uint8),
                caption="Gambar setelah pra-pemrosesan",
                use_container_width=True,
            )

        with gradcam_col2:
            st.image(
                overlay,
                caption=f"Peta perhatian model untuk prediksi '{predicted_info['label']}'",
                use_container_width=True,
            )

        st.info(
            "Peta panas Grad-CAM menunjukkan area daun yang paling memengaruhi "
            "keputusan model — warna hangat (merah/kuning) menandakan area "
            "dengan tingkat kepentingan lebih tinggi."
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("👆 Unggah gambar daun padi untuk memulai analisis.")

st.markdown(
    '<div class="footer-note">Rice Leaf Disease Classifier — EfficientNet-B0 + CBAM + Grad-CAM</div>',
    unsafe_allow_html=True,
)