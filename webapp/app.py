"""
Heritage Damage Assessment — Gradio web app
Classifier (EfficientNet-B4) + Detector (YOLOv8l) + Segmentor (MAnet/mit_b2, 3-phase progressive training)
Final: Crack IoU 96.23% | mIoU 85.67% | Dice 98.08%
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    _SAHI_OK = True
except ImportError:
    _SAHI_OK = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

CLS_CKPT  = MODELS_DIR / "classifier_best.pth"
SEG_CKPT  = MODELS_DIR / "segmentor_best.pth"
DET_CKPT  = MODELS_DIR / "detector_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# ---------------------------------------------------------------------------
# Preprocessing constants (ImageNet)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

CLS_SIZE = 380
SEG_SIZE = 512

cls_transform = transforms.Compose([
    transforms.Resize((CLS_SIZE, CLS_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

seg_transform = transforms.Compose([
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_classifier() -> nn.Module:
    m = tv_models.efficientnet_b4(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_feat, 2)
    return m


def _build_segmentor() -> nn.Module:
    return smp.MAnet(
        encoder_name="mit_b2",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


# ---------------------------------------------------------------------------
# Lazy model loading (cached on first call)
# ---------------------------------------------------------------------------
_models: dict = {}


def _load_models():
    if _models:
        return

    if not CLS_CKPT.exists():
        raise FileNotFoundError(f"Classifier weights not found: {CLS_CKPT}")
    if not SEG_CKPT.exists():
        raise FileNotFoundError(f"Segmentor weights not found: {SEG_CKPT}")
    if not DET_CKPT.exists():
        raise FileNotFoundError(f"Detector weights not found: {DET_CKPT}")

    # Classifier
    cls_model = _build_classifier()
    ckpt = torch.load(CLS_CKPT, map_location="cpu")
    cls_model.load_state_dict(ckpt["model_state"])
    cls_model.eval().to(DEVICE)
    _models["cls"] = cls_model

    # Segmentor
    seg_model = _build_segmentor()
    ckpt = torch.load(SEG_CKPT, map_location="cpu")
    seg_model.load_state_dict(ckpt["model_state"])
    seg_model.eval().to(DEVICE)
    _models["seg"] = seg_model

    # Detector — SAHI sliced inference when available, plain YOLO fallback
    if _SAHI_OK:
        _models["det"] = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(DET_CKPT),
            confidence_threshold=0.25,
            device=str(DEVICE),
        )
        _models["det_sahi"] = True
    else:
        _models["det"] = YOLO(str(DET_CKPT))
        _models["det_sahi"] = False


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Intact", "Cracked"]
CRACK_COLOR = (220, 50, 50)   # red overlay for segmentation mask


def _classify(pil_img: Image.Image) -> tuple[str, dict]:
    tensor = cls_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_models["cls"](tensor), dim=1)[0].cpu().numpy()
    label = CLASS_NAMES[int(probs.argmax())]
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(2)}
    return label, confidences


def _segment(pil_img: Image.Image) -> Image.Image:
    orig_w, orig_h = pil_img.size
    # LAB+CLAHE preprocessing (match training)
    img_arr = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_arr, cv2.COLOR_RGB2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    img_arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    pil_img = Image.fromarray(img_arr)
    tensor = seg_transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = _models["seg"](tensor)
    mask = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.70).astype(np.uint8)
    # Resize mask back to original resolution
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    # Colour overlay
    img_arr = np.array(pil_img.convert("RGB"))
    overlay = img_arr.copy()
    overlay[mask_full == 1] = (
        overlay[mask_full == 1] * 0.45
        + np.array(CRACK_COLOR) * 0.55
    ).astype(np.uint8)
    # Draw mask contours for cleaner boundary
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, CRACK_COLOR, 2)
    crack_pct = float(mask_full.mean() * 100)
    return Image.fromarray(overlay), crack_pct


def _detect(pil_img: Image.Image) -> tuple[Image.Image, int]:
    img_arr = np.array(pil_img.convert("RGB"))

    if _models.get("det_sahi"):
        w, h = pil_img.size
        # Use slices ~1/4 image area; full-image pass catches large regions too
        slice_size = max(256, min(512, min(w, h) // 2))
        result = get_sliced_prediction(
            pil_img,
            _models["det"],
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            perform_standard_pred=True,
            verbose=0,
        )
        annotated = img_arr.copy()
        for pred in result.object_prediction_list:
            x1, y1 = int(pred.bbox.minx), int(pred.bbox.miny)
            x2, y2 = int(pred.bbox.maxx), int(pred.bbox.maxy)
            conf = pred.score.value
            cv2.rectangle(annotated, (x1, y1), (x2, y2), CRACK_COLOR, 2)
            cv2.putText(
                annotated, f"crack {conf:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CRACK_COLOR, 1, cv2.LINE_AA,
            )
        return Image.fromarray(annotated), len(result.object_prediction_list)

    # Fallback: plain YOLO
    results = _models["det"].predict(img_arr, device=DEVICE, verbose=False, conf=0.25)
    annotated = cv2.cvtColor(results[0].plot(line_width=2), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), len(results[0].boxes)


# ---------------------------------------------------------------------------
# Main inference function (called by Gradio)
# ---------------------------------------------------------------------------

def predict(image: Image.Image):
    if image is None:
        return (
            "No image provided.",
            None,
            None,
            None,
        )

    _load_models()
    image = image.convert("RGB")

    # Classification
    cls_label, cls_conf = _classify(image)
    cls_text = (
        f"**{cls_label}**\n\n"
        f"Cracked: {cls_conf['Cracked']*100:.1f}%  |  "
        f"Intact: {cls_conf['Intact']*100:.1f}%"
    )

    # Detection
    det_img, n_boxes = _detect(image)
    det_caption = f"{n_boxes} crack region(s) detected" if n_boxes else "No cracks detected"

    # Segmentation
    seg_img, crack_pct = _segment(image)
    seg_caption = f"Crack coverage: {crack_pct:.1f}% of image area"

    return cls_text, cls_conf, det_img, det_caption, seg_img, seg_caption


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    ["examples/cracked_1.jpg"],
    ["examples/cracked_2.jpg"],
    ["examples/intact_1.jpg"],
]

HEADER = """
<div style="text-align:center; padding: 16px 0 8px 0;">
  <h1 style="font-size:2rem; margin-bottom:4px;">🏛 Albanian Heritage Damage Assessment</h1>
  <p style="color:#555; max-width:640px; margin:auto; line-height:1.5;">
    AI-powered structural damage detection for Albanian UNESCO World Heritage sites —
    <strong>Berat</strong>, <strong>Gjirokastër</strong>, and <strong>Butrint</strong>.
    Upload a historic stone or masonry surface image; the pipeline runs three specialised models in parallel.
  </p>
</div>
"""

BADGE_ROW = """
<div style="display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin: 8px 0 20px 0; font-size:0.85rem; color:#444;">
  <span>🔍 <strong>Classification</strong> — EfficientNet-B4 &nbsp;|&nbsp; 99.8% accuracy</span>
  <span>📦 <strong>Detection</strong> — YOLOv8s &nbsp;|&nbsp; mAP@50 = 96.7%</span>
  <span>🗺 <strong>Segmentation</strong> — U-Net/ResNet34 &nbsp;|&nbsp; mIoU = 83.6%</span>
</div>
"""

css = """
#analyse-btn { font-size: 1.1rem; height: 48px; }
.result-label { font-size: 1.4rem; font-weight: 700; }
footer { display: none !important; }
"""

OUTPUTS = lambda: [
    cls_out_text,
    cls_out_conf,
    det_out_img,
    det_out_caption,
    seg_out_img,
    seg_out_caption,
]

with gr.Blocks(
    title="Albanian Heritage Damage Assessment",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css=css,
) as demo:
    gr.HTML(HEADER)
    gr.HTML(BADGE_ROW)

    with gr.Row(equal_height=True):
        # ── Left column ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=280):
            inp = gr.Image(type="pil", label="Input image", height=320)
            run_btn = gr.Button("Analyse", variant="primary", elem_id="analyse-btn")
            gr.Examples(
                examples=[
                    ["examples/cracked_1.jpg"],
                    ["examples/cracked_2.jpg"],
                    ["examples/intact_1.jpg"],
                ],
                inputs=inp,
                label="Try an example",
            )

        # ── Right column ─────────────────────────────────────────────────────
        with gr.Column(scale=2):
            with gr.Tab("🔍 Classification"):
                cls_out_text = gr.Markdown(elem_classes=["result-label"])
                cls_out_conf = gr.Label(label="Confidence", num_top_classes=2)

            with gr.Tab("📦 Detection"):
                det_out_img     = gr.Image(label="Crack regions", height=360)
                det_out_caption = gr.Markdown()

            with gr.Tab("🗺 Segmentation"):
                seg_out_img     = gr.Image(label="Pixel-level crack mask", height=360)
                seg_out_caption = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=[inp],
        outputs=[cls_out_text, cls_out_conf, det_out_img, det_out_caption, seg_out_img, seg_out_caption],
    )

    gr.HTML(
        '<p style="text-align:center; color:#888; font-size:0.8rem; margin-top:24px;">'
        "Built for a Master's project in Computer Vision · Models trained on OmniCrack30k, "
        "COCO-Crack, MasonryCrack &amp; CrackForest datasets."
        "</p>"
    )


if __name__ == "__main__":
    demo.launch()
