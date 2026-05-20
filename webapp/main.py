"""
Heritage Damage Assessment — FastAPI backend
EfficientNet-B4 (classifier) + YOLOv8s (detector) + U-Net/ResNet34 (segmentor)
"""

from __future__ import annotations

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as tv_models
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
# Grad-CAM
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.eval()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return (alpha * heatmap + (1 - alpha) * image_rgb).astype(np.uint8)

# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).parent
MODELS_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

CLS_CKPT = MODELS_DIR / "classifier_best.pth"
SEG_CKPT = MODELS_DIR / "segmentor_best.pth"
DET_CKPT = MODELS_DIR / "detector_best.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

cls_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

seg_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _build_classifier() -> nn.Module:
    m = tv_models.efficientnet_b4(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m


def _build_segmentor() -> nn.Module:
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )


# ---------------------------------------------------------------------------
# Lazy model loading (thread-safe)
# ---------------------------------------------------------------------------
_models: dict = {}
_load_lock = threading.Lock()


def load_models():
    with _load_lock:
        if _models:
            return
        cls_model = _build_classifier()
        ckpt = torch.load(CLS_CKPT, map_location="cpu")
        cls_model.load_state_dict(ckpt["model_state"])
        cls_model.eval().to(DEVICE)
        _models["cls"] = cls_model
        _models["gradcam"] = GradCAM(cls_model, cls_model.features[-1])

        seg_model = _build_segmentor()
        ckpt = torch.load(SEG_CKPT, map_location="cpu")
        seg_model.load_state_dict(ckpt["model_state"])
        seg_model.eval().to(DEVICE)
        _models["seg"] = seg_model

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
CLASS_NAMES  = ["Intact", "Cracked"]
CRACK_COLOR  = (220, 50, 50)
EXECUTOR     = ThreadPoolExecutor(max_workers=3)


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _classify(img: Image.Image) -> dict:
    tensor = cls_transform(img).unsqueeze(0).to(DEVICE)
    # GradCAM needs gradients — do forward+backward first
    cam = _models["gradcam"](tensor)
    # Clean probability pass
    with torch.no_grad():
        probs = torch.softmax(_models["cls"](tensor), dim=1)[0].cpu().numpy()
    label = CLASS_NAMES[int(probs.argmax())]
    img_arr = np.array(img)
    cam_overlay = GradCAM.overlay(img_arr, cam)
    return {
        "label":       label,
        "cracked_pct": float(probs[1] * 100),
        "intact_pct":  float(probs[0] * 100),
        "cam_b64":     _pil_to_b64(Image.fromarray(cam_overlay)),
    }


def _segment(img: Image.Image) -> dict:
    orig_w, orig_h = img.size
    tensor = seg_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = _models["seg"](tensor)
    mask = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    mask_full = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    img_arr  = np.array(img)
    overlay  = img_arr.copy()
    overlay[mask_full == 1] = (
        overlay[mask_full == 1] * 0.4 + np.array(CRACK_COLOR) * 0.6
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, CRACK_COLOR, 2)

    return {
        "image_b64": _pil_to_b64(Image.fromarray(overlay)),
        "crack_pct": float(mask_full.mean() * 100),
    }


def _draw_boxes(arr: np.ndarray, preds: list, color: tuple) -> np.ndarray:
    out = arr.copy()
    for x1, y1, x2, y2, conf in preds:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"Crack {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _detect(img: Image.Image) -> dict:
    arr = np.array(img)
    sensitive = False

    if _models.get("det_sahi"):
        w, h = img.size
        slice_size = max(256, min(512, min(w, h) // 2))
        result = get_sliced_prediction(
            img, _models["det"],
            slice_height=slice_size, slice_width=slice_size,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2,
            perform_standard_pred=True,
            verbose=0,
        )
        preds = [
            (int(p.bbox.minx), int(p.bbox.miny),
             int(p.bbox.maxx), int(p.bbox.maxy),
             p.score.value)
            for p in result.object_prediction_list
        ]
        # Sensitive fallback: re-run SAHI at lower threshold if nothing found
        if not preds:
            _models["det"].confidence_threshold = 0.10
            result2 = get_sliced_prediction(
                img, _models["det"],
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2,
                perform_standard_pred=True,
                verbose=0,
            )
            _models["det"].confidence_threshold = 0.25
            preds = [
                (int(p.bbox.minx), int(p.bbox.miny),
                 int(p.bbox.maxx), int(p.bbox.maxy),
                 p.score.value)
                for p in result2.object_prediction_list
            ]
            sensitive = True
    else:
        results = _models["det"].predict(arr, device=DEVICE, verbose=False, conf=0.25)
        boxes = results[0].boxes
        if len(boxes) == 0:
            results = _models["det"].predict(arr, device=DEVICE, verbose=False, conf=0.10)
            boxes = results[0].boxes
            sensitive = True
        preds = [
            (*box.xyxy[0].cpu().int().tolist(), float(box.conf[0]))
            for box in boxes
        ]

    color = (255, 140, 0) if not sensitive else (100, 160, 255)
    out = _draw_boxes(arr, preds, color)

    return {
        "image_b64": _pil_to_b64(Image.fromarray(out)),
        "n_boxes":   len(preds),
        "sensitive": sensitive,
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Heritage Damage Assessment")

app.mount("/examples", StaticFiles(directory=str(ROOT / "examples")), name="examples")
app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/analyze")
def analyze_page():
    return FileResponse(STATIC_DIR / "app.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    load_models()

    # Run all 3 models concurrently
    fut_cls = EXECUTOR.submit(_classify, img)
    fut_seg = EXECUTOR.submit(_segment, img)
    fut_det = EXECUTOR.submit(_detect,  img)

    cls_res = fut_cls.result()
    seg_res = fut_seg.result()
    det_res = fut_det.result()

    return JSONResponse({
        "classification": cls_res,
        "segmentation":   seg_res,
        "detection":      det_res,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
