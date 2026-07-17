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
)
from models import build_model
from utils import preprocess_image, predict_image, get_gradcam, generate_gradcam_overlay


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Rice Leaf Disease Classifier",
    page_icon="🌾",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached resources: model + Grad-CAM object
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
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
st.sidebar.title("⚙️ Settings")

device_option = st.sidebar.radio(
    "Device",
    options=["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model checkpoint")

uploaded_ckpt = st.sidebar.file_uploader(
    "Upload a .pth checkpoint (optional)",
    type=["pth", "pt"],
    help="If you don't have one on disk yet, upload your trained weights here.",
)

checkpoint_path = DEFAULT_CHECKPOINT_PATH

if uploaded_ckpt is not None:
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    saved_ckpt_path = os.path.join(WEIGHTS_DIR, uploaded_ckpt.name)
    with open(saved_ckpt_path, "wb") as f:
        f.write(uploaded_ckpt.getbuffer())
    checkpoint_path = saved_ckpt_path
    st.sidebar.success(f"Using uploaded checkpoint: {uploaded_ckpt.name}")
elif os.path.exists(DEFAULT_CHECKPOINT_PATH):
    st.sidebar.info(f"Using default checkpoint:\n`{os.path.basename(DEFAULT_CHECKPOINT_PATH)}`")
else:
    st.sidebar.warning(
        "No checkpoint found in weights/ and none uploaded. "
        "The model will run with ImageNet-pretrained weights only "
        "(predictions will be meaningless until you provide a trained checkpoint)."
    )
    checkpoint_path = None

st.sidebar.markdown("---")
st.sidebar.subheader("Classes")
st.sidebar.write(", ".join(CLASS_NAMES))


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🌾 Rice Leaf Disease Classifier")
st.caption(
    "EfficientNet-B0 backbone + CBAM attention, with Grad-CAM visual "
    "explanation of the model's decision."
)

uploaded_image = st.file_uploader(
    "Upload a rice leaf image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

run_button_placeholder = st.empty()

if uploaded_image is not None:
    pil_image = Image.open(uploaded_image).convert("RGB")

    col_input, col_result = st.columns(2)

    with col_input:
        st.subheader("Input Image")
        st.image(pil_image, use_container_width=True)

    analyze_clicked = run_button_placeholder.button("🔍 Analyze Image", type="primary")

    if analyze_clicked:
        with st.spinner("Loading model and running inference..."):
            model = load_model(checkpoint_path, device_option)
            cam = load_cam(model)

            input_tensor = preprocess_image(pil_image, image_size=IMAGE_SIZE)

            predicted_name, confidence, probabilities = predict_image(
                model, input_tensor, CLASS_NAMES, device=device_option
            )

            overlay, original_image = generate_gradcam_overlay(
                cam, input_tensor.to(device_option)
            )

        with col_result:
            st.subheader("Prediction")
            if predicted_name.lower() == "healthy":
                st.success(f"**{predicted_name}** — {confidence:.2f}% confidence")
            else:
                st.error(f"**{predicted_name}** — {confidence:.2f}% confidence")

            st.write("**Class probabilities:**")
            for class_name, prob in sorted(
                probabilities.items(), key=lambda kv: kv[1], reverse=True
            ):
                st.progress(min(int(prob), 100), text=f"{class_name}: {prob:.2f}%")

        st.markdown("---")
        st.subheader("Grad-CAM Explanation")
        gradcam_col1, gradcam_col2 = st.columns(2)

        with gradcam_col1:
            st.image(
                (original_image * 255).astype(np.uint8),
                caption="Preprocessed input",
                use_container_width=True,
            )

        with gradcam_col2:
            st.image(
                overlay,
                caption=f"Grad-CAM overlay (focus behind '{predicted_name}' prediction)",
                use_container_width=True,
            )

        st.info(
            "The Grad-CAM heatmap highlights the regions of the leaf that "
            "most influenced the model's prediction — warmer colors (red/"
            "yellow) indicate areas of higher importance."
        )
else:
    st.info("👆 Upload a rice leaf image to get started.")
