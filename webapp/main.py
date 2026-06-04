"""
Heritage Damage Assessment — FastAPI backend
EfficientNet-B4 (classifier) + YOLOv8s (detector) + MAnet/mit_b2 (segmentor)
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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

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
    return smp.MAnet(
        encoder_name="mit_b2",
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
        # Handle both old format (dict with "model_state") and new format (state_dict directly)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        seg_model.load_state_dict(state)
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
EXECUTOR     = ThreadPoolExecutor(max_workers=6)


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
    """Segment cracks using MAnet model. Returns segmentation overlay and crack percentage."""
    img_arr = np.array(img.convert("RGB"))
    h_orig, w_orig = img_arr.shape[:2]

    # Preprocess: CLAHE + resize + normalize
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    enhanced = _CLAHE.apply(gray)
    img_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(img_enhanced, (384, 384))
    img_tensor = seg_transform(Image.fromarray(img_resized)).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = _models["seg"](img_tensor)

    # Apply threshold (0.3 from training)
    seg_mask = torch.sigmoid(logits[0, 0]).cpu().numpy()
    seg_binary = (seg_mask > 0.3).astype(np.uint8) * 255

    # Resize back to original size
    seg_binary_orig = cv2.resize(seg_binary, (w_orig, h_orig))

    # Create red overlay on original image
    result = img_arr.copy().astype(np.float32)
    mask_3ch = np.stack([seg_binary_orig] * 3, axis=2) / 255.0
    overlay_color = np.array(CRACK_COLOR, dtype=np.float32)
    result = result * (1 - 0.4 * mask_3ch) + overlay_color * 0.4 * mask_3ch
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Calculate crack percentage
    crack_pixels = np.sum(seg_binary_orig > 0)
    total_pixels = h_orig * w_orig
    crack_pct = (crack_pixels / total_pixels) * 100.0

    return {
        "image_b64": _pil_to_b64(Image.fromarray(result)),
        "crack_pct": crack_pct,
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
        try:
            results = _models["det"].predict(arr, device=DEVICE, verbose=False, conf=0.25)
        except AttributeError:
            # Fusion error: disable model fusion
            _models["det"].fuse = lambda: _models["det"]
            results = _models["det"].predict(arr, device=DEVICE, verbose=False, conf=0.25)
        boxes = results[0].boxes
        if len(boxes) == 0:
            try:
                results = _models["det"].predict(arr, device=DEVICE, verbose=False, conf=0.10)
            except AttributeError:
                _models["det"].fuse = lambda: _models["det"]
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
        "predictions": preds,  # List of (x1, y1, x2, y2, conf) tuples
    }


def _surface_analysis(img: Image.Image) -> dict:
    arr     = np.array(img)
    orig_w, orig_h = img.size

    small = cv2.resize(arr, (256, 256))
    gray  = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Gabor energy: 6 orientations × 3 wavelengths (matches nb09 filter bank)
    gabor_map = np.zeros((256, 256), dtype=np.float32)
    for ti in range(6):
        theta = ti * np.pi / 6
        for lam in (8, 12, 20):
            kern = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                       lambd=float(lam), gamma=0.5, psi=0,
                                       ktype=cv2.CV_32F)
            gabor_map = np.maximum(gabor_map,
                                   np.abs(cv2.filter2D(gray, cv2.CV_32F, kern)))
    gabor_norm = (gabor_map - gabor_map.min()) / (gabor_map.max() - gabor_map.min() + 1e-8)

    # Local complexity via sliding-window Laplacian std (proxy for entropy)
    blurred   = cv2.GaussianBlur(gray, (9, 9), 2.0)
    lap       = cv2.Laplacian(blurred, cv2.CV_32F)
    lap_abs   = np.abs(lap)
    k         = 11
    mean_sq   = cv2.boxFilter(lap_abs ** 2, -1, (k, k))
    mean_     = cv2.boxFilter(lap_abs,      -1, (k, k))
    local_std = np.sqrt(np.maximum(mean_sq - mean_ ** 2, 0))
    comp_norm = (local_std - local_std.min()) / (local_std.max() - local_std.min() + 1e-8)

    # Texture score: Laplacian variance + mean Gabor energy, each normalised by
    # empirical calibration from nb09 degraded heritage images
    lap_var    = float(np.var(lap))
    gabor_mean = float(gabor_map.mean())
    lap_s  = float(np.clip(lap_var   / 0.015, 0, 1))
    gab_s  = float(np.clip(gabor_mean / 0.06,  0, 1))
    score  = round((0.5 * lap_s + 0.5 * gab_s) * 100, 1)

    def _heatmap_b64(norm_map: np.ndarray) -> str:
        h = cv2.applyColorMap((norm_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        h = cv2.cvtColor(cv2.resize(h, (orig_w, orig_h)), cv2.COLOR_BGR2RGB)
        return _pil_to_b64(Image.fromarray(h))

    return {
        "gabor_b64":     _heatmap_b64(gabor_norm),
        "entropy_b64":   _heatmap_b64(comp_norm),
        "texture_score": score,
    }


# ---------------------------------------------------------------------------
# Preprocessing filters
# ---------------------------------------------------------------------------
FILTER_NAMES = [
    "none", "clahe", "gamma_correct",
    "sobel_overlay", "canny_overlay", "log_enhanced",
    "fft_highpass", "fft_lowpass",
    "crack_extract", "tophat_morph", "gradient_morph",
    "bilateral_denoise", "median_denoise",
]

FILTER_LABELS = {
    "none":             "Original",
    "clahe":            "CLAHE",
    "gamma_correct":    "Gamma Correct",
    "sobel_overlay":    "Sobel Overlay",
    "canny_overlay":    "Canny Overlay",
    "log_enhanced":     "LoG Enhanced",
    "fft_highpass":     "FFT High-Pass",
    "fft_lowpass":      "FFT Low-Pass",
    "crack_extract":    "Crack Extract",
    "tophat_morph":     "Top-Hat",
    "gradient_morph":   "Morph Gradient",
    "bilateral_denoise":"Bilateral Denoise",
    "median_denoise":   "Median Denoise",
}

FILTER_DESCRIPTIONS = {
    "none":             "No preprocessing — raw input to models",
    "clahe":            "Local contrast enhancement in LAB L-channel",
    "gamma_correct":    "Gamma γ=0.6 — lifts shadow detail",
    "sobel_overlay":    "Sobel gradient magnitude blended onto image",
    "canny_overlay":    "Canny edge contours blended onto original",
    "log_enhanced":     "Laplacian of Gaussian (Marr-Hildreth) blend",
    "fft_highpass":     "Gaussian high-pass in 2D Fourier domain",
    "fft_lowpass":      "Gaussian low-pass — smooths high-freq noise",
    "crack_extract":    "Bilateral + morphological black-hat transform",
    "tophat_morph":     "White top-hat — bright deposits on dark stone",
    "gradient_morph":   "Morphological gradient — object boundaries",
    "bilateral_denoise":"Strong bilateral — noise reduction, edge-safe",
    "median_denoise":   "Median filter — removes salt-and-pepper noise",
}

FILTER_CATEGORIES = {
    "none":             "Enhancement",
    "clahe":            "Enhancement",
    "gamma_correct":    "Enhancement",
    "sobel_overlay":    "Edge & Structure",
    "canny_overlay":    "Edge & Structure",
    "log_enhanced":     "Edge & Structure",
    "fft_highpass":     "Frequency",
    "fft_lowpass":      "Frequency",
    "crack_extract":    "Morphology",
    "tophat_morph":     "Morphology",
    "gradient_morph":   "Morphology",
    "bilateral_denoise":"Noise Reduction",
    "median_denoise":   "Noise Reduction",
}


def _apply_filter(img: Image.Image, filter_name: str) -> Image.Image:
    if filter_name == "none":
        return img

    arr = np.array(img)

    if filter_name == "clahe":
        # Contrast Limited Adaptive Histogram Equalization on LAB L-channel
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "crack_extract":
        # Bilateral denoise → black-hat morphology (extracts dark thin cracks) → CLAHE
        denoised = cv2.bilateralFilter(arr, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_3ch = cv2.cvtColor(blackhat, cv2.COLOR_GRAY2RGB)
        combined = cv2.addWeighted(denoised, 0.6, blackhat_3ch, 0.4, 0)
        lab = cv2.cvtColor(combined, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "fft_highpass":
        # 2D DFT → Gaussian high-pass mask → IDFT → blend with original
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)
        h, w = L.shape
        cy, cx = h // 2, w // 2
        fshift = np.fft.fftshift(np.fft.fft2(L))
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        radius = min(h, w) / 8.0
        hp_mask = 1.0 - np.exp(-dist ** 2 / (2 * radius ** 2))  # Gaussian high-pass
        L_hp = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * hp_mask)))
        L_hp_norm = (L_hp / (L_hp.max() + 1e-8) * 80).astype(np.float32)
        lab[:, :, 0] = np.clip(L + L_hp_norm, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "canny_overlay":
        # Canny edge detection (bilateral pre-smooth) blended at 20% onto original
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
        edges = cv2.Canny(smooth, threshold1=40, threshold2=120)
        edge_color = np.zeros_like(arr)
        edge_color[edges > 0] = [220, 90, 20]
        blended = cv2.addWeighted(arr, 0.82, edge_color, 0.18, 0)
        lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "log_enhanced":
        # Laplacian of Gaussian (Marr-Hildreth): LoG magnitude blended as brightness boost
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.2)
        log_resp = cv2.Laplacian(blurred, cv2.CV_32F)
        log_abs = np.abs(log_resp)
        log_norm = (log_abs / (log_abs.max() + 1e-8) * 70).astype(np.uint8)
        boost = cv2.cvtColor(log_norm, cv2.COLOR_GRAY2RGB)
        blended = cv2.addWeighted(arr, 0.85, boost, 0.15, 0)
        lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "gamma_correct":
        # Gamma correction γ=0.6 lifts shadow detail in dark stone surfaces
        table = np.array([(i / 255.0) ** 0.6 * 255 for i in range(256)], dtype=np.uint8)
        out = cv2.LUT(arr, table)

    elif filter_name == "sobel_overlay":
        # Sobel gradient magnitude (both axes) blended as warm-tinted edge overlay
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag_norm = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
        edge_color = np.zeros_like(arr)
        edge_color[:, :, 0] = mag_norm
        edge_color[:, :, 1] = (mag_norm * 0.4).astype(np.uint8)
        blended = cv2.addWeighted(arr, 0.72, edge_color, 0.28, 0)
        lab = cv2.cvtColor(blended, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "fft_lowpass":
        # Gaussian low-pass in frequency domain — suppresses high-freq noise
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype(np.float32)
        h, w = L.shape
        cy, cx = h // 2, w // 2
        fshift = np.fft.fftshift(np.fft.fft2(L))
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        radius = min(h, w) / 5.0
        lp_mask = np.exp(-dist ** 2 / (2 * radius ** 2))
        L_lp = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * lp_mask)))
        lab[:, :, 0] = np.clip(L_lp / (L_lp.max() + 1e-8) * 255, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "tophat_morph":
        # White top-hat extracts small bright features (efflorescence, salt deposits)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_3ch = cv2.cvtColor(tophat, cv2.COLOR_GRAY2RGB)
        combined = cv2.addWeighted(arr, 0.6, tophat_3ch, 0.4, 0)
        lab = cv2.cvtColor(combined, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    elif filter_name == "gradient_morph":
        # Morphological gradient (dilation − erosion) highlights region boundaries
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        grad_eq = cv2.equalizeHist(grad)
        grad_3ch = cv2.cvtColor(grad_eq, cv2.COLOR_GRAY2RGB)
        out = cv2.addWeighted(arr, 0.65, grad_3ch, 0.35, 0)

    elif filter_name == "bilateral_denoise":
        # Strong bilateral filter — noise reduction while preserving edges
        out = cv2.bilateralFilter(arr, d=15, sigmaColor=100, sigmaSpace=100)

    elif filter_name == "median_denoise":
        # Median filter — removes salt-and-pepper noise, preserves edges
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        out = np.stack([cv2.medianBlur(c, 7) for c in [r, g, b]], axis=2)

    else:
        return img

    return Image.fromarray(out.astype(np.uint8))


def _apply_pipeline(img: Image.Image, pipeline_str: str) -> Image.Image:
    """Apply an ordered comma-separated filter pipeline to an image."""
    names = [n.strip() for n in pipeline_str.split(",") if n.strip() and n.strip() != "none"]
    for name in names:
        if name in FILTER_NAMES:
            img = _apply_filter(img, name)
    return img


def _pipeline_label(pipeline_str: str) -> str:
    names = [n.strip() for n in pipeline_str.split(",") if n.strip() and n.strip() != "none"]
    labels = [FILTER_LABELS[n] for n in names if n in FILTER_LABELS]
    return " → ".join(labels) if labels else "Original"


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Heritage Damage Assessment")

app.mount("/examples", StaticFiles(directory=str(ROOT / "examples")), name="examples")
app.mount("/static",  StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/about")
def about_page():
    return FileResponse(STATIC_DIR / "about.html")


@app.get("/analyze")
def analyze_page():
    return FileResponse(STATIC_DIR / "app.html")


@app.post("/filter-preview")
async def filter_preview(file: UploadFile = File(...)):
    """Return all filtered versions of the uploaded image as base64 thumbnails."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Thumbnail for fast preview (300px wide)
    thumb_w = 300
    ratio = thumb_w / img.width
    thumb_h = int(img.height * ratio)
    thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)

    previews = {}
    for name in FILTER_NAMES:
        filtered = _apply_filter(thumb, name)
        previews[name] = {
            "label":       FILTER_LABELS[name],
            "description": FILTER_DESCRIPTIONS[name],
            "category":    FILTER_CATEGORIES[name],
            "image_b64":   _pil_to_b64(filtered),
        }

    return JSONResponse({"filters": previews})


@app.post("/predict")
async def predict(file: UploadFile = File(...), filter_name: str = Form("none")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    invalid = [n.strip() for n in filter_name.split(",") if n.strip() and n.strip() not in FILTER_NAMES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown filter(s): {invalid}")

    raw = await file.read()
    img_orig = Image.open(io.BytesIO(raw)).convert("RGB")

    load_models()

    # Run models on ORIGINAL image (no preprocessing filters)
    fut_cls  = EXECUTOR.submit(_classify, img_orig)
    fut_det  = EXECUTOR.submit(_detect, img_orig)
    fut_seg  = EXECUTOR.submit(_segment, img_orig)
    fut_surf = EXECUTOR.submit(_surface_analysis, img_orig)

    cls_res  = fut_cls.result()
    det_res  = fut_det.result()
    seg_res  = fut_seg.result()
    surf_res = fut_surf.result()

    return JSONResponse({
        "classification": cls_res,
        "segmentation":   seg_res,
        "detection":      det_res,
        "surface":        surf_res,
        "filter_applied": _pipeline_label(filter_name),
    })


def _ssim_map(g1: np.ndarray, g2: np.ndarray) -> tuple[float, np.ndarray]:
    """SSIM score + per-pixel change map (1-SSIM) for two uint8 grayscale images."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    f1, f2 = g1.astype(np.float32), g2.astype(np.float32)
    k = (cv2.getGaussianKernel(11, 1.5) @ cv2.getGaussianKernel(11, 1.5).T).astype(np.float32)
    mu1, mu2 = cv2.filter2D(f1, -1, k), cv2.filter2D(f2, -1, k)
    mu1_sq, mu2_sq, mu12 = mu1**2, mu2**2, mu1*mu2
    s1  = cv2.filter2D(f1*f1, -1, k) - mu1_sq
    s2  = cv2.filter2D(f2*f2, -1, k) - mu2_sq
    s12 = cv2.filter2D(f1*f2, -1, k) - mu12
    ssim = ((2*mu12 + C1) * (2*s12 + C2)) / ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(ssim.mean()), np.clip(1.0 - ssim, 0, 1)


@app.post("/compare")
async def compare_images(before: UploadFile = File(...), after: UploadFile = File(...), filter_name: str = Form("none")):
    if not before.content_type.startswith("image/") or not after.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Both files must be images")
    invalid = [n.strip() for n in filter_name.split(",") if n.strip() and n.strip() not in FILTER_NAMES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown filter(s): {invalid}")

    img_b = Image.open(io.BytesIO(await before.read())).convert("RGB")
    img_a = Image.open(io.BytesIO(await after.read())).convert("RGB")
    img_b = _apply_pipeline(img_b, filter_name)
    img_a = _apply_pipeline(img_a, filter_name)

    load_models()

    # Align to common size for SSIM
    W = min(img_b.width,  img_a.width,  512)
    H = min(img_b.height, img_a.height, 512)
    ib = img_b.resize((W, H), Image.LANCZOS)
    ia = img_a.resize((W, H), Image.LANCZOS)

    gray_b = cv2.cvtColor(np.array(ib), cv2.COLOR_RGB2GRAY)
    gray_a = cv2.cvtColor(np.array(ia), cv2.COLOR_RGB2GRAY)
    ssim_score, change_raw = _ssim_map(gray_b, gray_a)

    # Colorise change map
    change_norm = (change_raw - change_raw.min()) / (change_raw.max() - change_raw.min() + 1e-8)
    change_vis  = cv2.cvtColor(
        cv2.applyColorMap((change_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB)
    change_b64 = _pil_to_b64(Image.fromarray(change_vis))

    # Run models on both images concurrently
    fut_seg_b  = EXECUTOR.submit(_segment,          img_b)
    fut_seg_a  = EXECUTOR.submit(_segment,          img_a)
    fut_surf_b = EXECUTOR.submit(_surface_analysis, img_b)
    fut_surf_a = EXECUTOR.submit(_surface_analysis, img_a)
    fut_cls_a  = EXECUTOR.submit(_classify,         img_a)

    seg_b, seg_a   = fut_seg_b.result(), fut_seg_a.result()
    surf_b, surf_a = fut_surf_b.result(), fut_surf_a.result()
    cls_a          = fut_cls_a.result()

    return JSONResponse({
        "change_map_b64": change_b64,
        "ssim_score":     round(ssim_score, 3),
        "ssim_change":    round(1.0 - ssim_score, 3),
        "filter_applied": _pipeline_label(filter_name),
        "before": {
            "crack_pct":     seg_b["crack_pct"],
            "seg_b64":       seg_b["image_b64"],
            "texture_score": surf_b["texture_score"],
        },
        "after": {
            "crack_pct":     seg_a["crack_pct"],
            "seg_b64":       seg_a["image_b64"],
            "texture_score": surf_a["texture_score"],
            "label":         cls_a["label"],
            "cracked_pct":   cls_a["cracked_pct"],
            "cam_b64":       cls_a["cam_b64"],
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
