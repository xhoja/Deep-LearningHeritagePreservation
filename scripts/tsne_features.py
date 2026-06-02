"""
t-SNE feature visualization for EfficientNet-B4 heritage damage classifier.
Extracts 1792-dim GAP features, projects to 2D, saves plot + CSV.
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Ellipse
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import transforms
import torchvision.models as tv_models
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT   = ROOT / "checkpoints/classifier/best.pth"
DOMAIN_DIRS  = {
    "HistoricalCrack": ROOT / "data/historical_crack",   # walks all subdirs
    "CrackForest":     ROOT / "data/crackforest/images",
    "Masonry":         ROOT / "data/masonry/images",
}
OUT_PLOT = ROOT / "outputs/tsne_efficientnet_b4.png"
OUT_CSV  = ROOT / "outputs/tsne_efficientnet_b4.csv"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_PER_DOMAIN = 100
SEED           = 42
DEVICE         = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ── style ────────────────────────────────────────────────────────────────────
BG_COLOR  = "#f3edd8"
DOMAIN_STYLE = {
    "HistoricalCrack": dict(color="#2d7a20", marker="o", label="HistoricalCrack"),
    "CrackForest":     dict(color="#c87010", marker="^", label="CrackForest"),
    "Masonry":         dict(color="#b03a20", marker="s", label="Masonry"),
}

# ─────────────────────────────────────────────────────────────────────────────

def collect_images(directory: Path, max_n: int, rng: random.Random) -> list[Path]:
    imgs = [p for p in directory.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    rng.shuffle(imgs)
    return imgs[:max_n]


def build_model(checkpoint: Path) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]

    num_classes = state["classifier.1.weight"].shape[0]

    model = tv_models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state)

    # drop classifier → 1792-dim flattened GAP vector
    model.classifier = nn.Identity()
    model.eval()
    return model.to(DEVICE)


def build_transform(img_size: int = 380) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(model: nn.Module, paths: list[Path], tfm: transforms.Compose) -> np.ndarray:
    feats = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        x = tfm(img).unsqueeze(0).to(DEVICE)
        f = model(x).squeeze(0).cpu().numpy()
        feats.append(f)
    return np.stack(feats) if feats else np.empty((0, 1792))


def fit_ellipse(ax: plt.Axes, xy: np.ndarray, color: str, n_std: float = 1.8):
    """Draw dashed covariance ellipse around a cluster."""
    if len(xy) < 3:
        return
    mean = xy.mean(0)
    cov  = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h  = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(mean, width=w, height=h, angle=angle,
                  edgecolor=color, facecolor="none",
                  linestyle="--", linewidth=1.4, alpha=0.75)
    ax.add_patch(ell)


def main():
    rng = random.Random(SEED)
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading model …")
    model = build_model(CHECKPOINT)
    tfm   = build_transform(img_size=380)

    all_feats, all_labels = [], []

    for domain, directory in DOMAIN_DIRS.items():
        paths = collect_images(directory, MAX_PER_DOMAIN, rng)
        print(f"  {domain}: {len(paths)} images")
        feats = extract_features(model, paths, tfm)
        all_feats.append(feats)
        all_labels.extend([domain] * len(feats))

    X      = np.concatenate(all_feats, axis=0)
    labels = np.array(all_labels)
    print(f"\nFeature matrix: {X.shape}")

    print("Running t-SNE …")
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
    Z    = tsne.fit_transform(X)

    # save CSV
    df = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "domain": labels})
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved CSV → {OUT_CSV}")

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    for domain, style in DOMAIN_STYLE.items():
        mask = labels == domain
        xy   = Z[mask]
        ax.scatter(xy[:, 0], xy[:, 1],
                   c=style["color"], marker=style["marker"],
                   s=38, alpha=0.82, linewidths=0.3,
                   edgecolors="white", label=style["label"], zorder=3)
        fit_ellipse(ax, xy, style["color"])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper right", framealpha=0.55,
        facecolor=BG_COLOR, edgecolor="#888",
        fontsize=9, markerscale=1.3,
    )

    ax.set_title(
        "t-SNE · EfficientNet-B4 features (1792-dim)",
        fontsize=11, pad=10, color="#333",
    )

    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"Saved plot → {OUT_PLOT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
