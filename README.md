# Rice Leaf Disease Classifier (EfficientNet-B0 + CBAM + Grad-CAM)

A Streamlit app that lets a user upload a rice leaf photo, runs it through
an EfficientNet-B0 backbone with a CBAM attention block, and returns the
predicted disease class along with a Grad-CAM heatmap explaining *why*.

## Project structure

```
rice_disease_app/
├── app.py                     # Streamlit UI — the entry point
├── config.py                  # All constants (paths, classes, image size, CBAM params)
├── requirements.txt
├── models/
│   ├── __init__.py
│   ├── cbam.py                # ChannelAttention, SpatialAttention, CBAM
│   └── efficientnet_cbam.py   # EfficientNetB0_CBAM model + build_model() loader
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py       # inference transform, denormalize_image
│   ├── predict.py             # predict_image() -> class, confidence, probs
│   └── gradcam_utils.py       # GradCAM wrapper + overlay generation
└── weights/
    └── (put your trained .pth checkpoint here)
```

## Setup

```bash
cd rice_disease_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Add your trained model

Copy your checkpoint (the file saved by `torch.save(model.state_dict(), SAVE_PATH)`
in the training notebook) into the `weights/` folder, named:

```
ProtoTypeEfficentNetB0CBAMV2Extra.pth
```

(or change `DEFAULT_CHECKPOINT_NAME` in `config.py` to match your filename).
You can also skip this and instead upload the checkpoint file directly from
the app's sidebar at runtime.

## Update class names

`config.py` has:

```python
CLASS_NAMES = ["BrownSpot", "Healthy", "LeafBlast", "LeafBlight"]
```

This **must** match the alphabetical folder order `ImageFolder` used during
training (i.e. `train_dataset.class_to_idx` from your notebook). Update the
list if your dataset's class names/order differ.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`),
upload a leaf image, and click **Analyze Image**. You'll get:

- The predicted disease class + confidence
- A per-class probability breakdown
- A side-by-side Grad-CAM heatmap showing which regions of the leaf drove
  the prediction

## Notes

- GPU is used automatically if available and selected in the sidebar;
  otherwise it falls back to CPU.
- Grad-CAM targets the last convolutional block of the EfficientNet feature
  extractor (`model.features[-1]`), matching the notebook's original setup.
- If no checkpoint is found/uploaded, the model still runs (using only
  ImageNet-pretrained backbone weights) so you can verify the UI works, but
  predictions won't be meaningful until you supply your trained weights.
