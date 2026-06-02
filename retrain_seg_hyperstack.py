#!/usr/bin/env python3
"""
Crack Segmentation — MAnet + mit_b2
Hyperstack training script (3-phase progressive training)

Usage:
    python retrain_seg_hyperstack.py [--phase 1|2|3] [--resume]

Setup:
    1. SSH into Hyperstack instance
    2. git clone https://github.com/xhoja/Deep-LearningHeritagePreservation.git
    3. cd Deep-LearningHeritagePreservation
    4. pip install -q torch torchvision segmentation-models-pytorch albumentations timm scikit-learn tqdm
    5. python retrain_seg_hyperstack.py
"""

import argparse
import json
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

REPO_DIR = Path(__file__).parent
DATA_DIR = REPO_DIR / 'data'
CKPT_DIR = REPO_DIR / 'checkpoints' / 'segmentor_v4'
PLOTS_DIR = REPO_DIR / 'plots' / 'segmentor_v4'

CKPT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ENCODER = 'mit_b2'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ============================================================================
# DATASET
# ============================================================================

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
IMG_DIR_NAMES = {'images', 'img', 'image', 'jpegimages', 'rgb', 'data'}
MASK_DIR_NAMES = {'masks', 'mask', 'labels', 'annotations', 'gt', 'ground_truth', 'seg'}

def find_dir(root, dir_names):
    root_p = Path(root)
    for d in root_p.rglob('*'):
        if d.is_dir() and d.name.lower() in dir_names:
            return d
    return None

def find_pairs(root):
    root_p = Path(root)
    img_dir = find_dir(root, IMG_DIR_NAMES) or root_p
    mask_dir = find_dir(root, MASK_DIR_NAMES) or root_p

    pairs = []
    for img in sorted(img_dir.rglob('*')):
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            stem = img.stem
            mask = mask_dir / f"{stem}.png"
            if not mask.exists():
                mask = mask_dir / f"{stem}.jpg"
            if mask.exists():
                pairs.append((img, mask))
    return pairs

def load_dataset():
    print(f"Loading from {DATA_DIR}...")
    all_pairs = []
    for dataset_dir in sorted(DATA_DIR.iterdir()):
        if dataset_dir.is_dir():
            pairs = find_pairs(dataset_dir)
            all_pairs.extend(pairs)
            print(f"  {dataset_dir.name:30s} {len(pairs):4d} pairs")

    print(f"Total: {len(all_pairs)} pairs")

    random.seed(42)
    train_p, temp_p = train_test_split(all_pairs, train_size=0.7, random_state=42)
    val_p, test_p = train_test_split(temp_p, train_size=0.5, random_state=42)

    print(f"Split: train {len(train_p)} | val {len(val_p)} | test {len(test_p)}")
    return train_p, val_p, test_p

# ============================================================================
# AUGMENTATIONS
# ============================================================================

def get_seg_transforms(split, img_size=384):
    if split == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

# ============================================================================
# DATASET CLASS
# ============================================================================

_CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

class CrackSegDataset(Dataset):
    def __init__(self, pairs, split='train', img_size=384, transform=None):
        self.pairs = pairs
        self.split = split
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            return self[random.randint(0, len(self) - 1)]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _CLAHE.apply(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']

        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

def make_loaders(train_pairs, val_pairs, test_pairs, img_size, batch_size, batch_eval=4):
    train_ds = CrackSegDataset(train_pairs, split='train', img_size=img_size,
                               transform=get_seg_transforms('train', img_size))
    val_ds = CrackSegDataset(val_pairs, split='val', img_size=img_size,
                             transform=get_seg_transforms('val', img_size))
    test_ds = CrackSegDataset(test_pairs, split='test', img_size=img_size,
                              transform=get_seg_transforms('val', img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, test_ds

# ============================================================================
# MODEL
# ============================================================================

def build_model(encoder=ENCODER):
    return smp.MAnet(
        encoder_name=encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
        activation=None,
    )

# ============================================================================
# LOSS & METRICS
# ============================================================================

tversky_loss = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, from_logits=True)
lovasz_loss = smp.losses.LovaszLoss(mode='binary', per_image=False, from_logits=True)

def combined_loss(pred, target):
    return 0.5 * tversky_loss(pred, target) + 0.5 * lovasz_loss(pred, target)

def compute_seg_metrics(pred_logits, target, threshold=0.5):
    pred_binary = (torch.sigmoid(pred_logits) > threshold).long()
    tp, fp, fn, tn = smp.metrics.get_stats(pred_binary, target.long(), mode='binary')
    return {
        'iou': smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item(),
        'dice': smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item(),
    }

def run_epoch(model, loader, optimizer, scaler, train=True):
    model.train() if train else model.eval()
    losses, ious = [], []

    for img, mask in tqdm(loader, desc=f'{"Train" if train else "Val":5s}', disable=False):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        with autocast('cuda'):
            pred = model(img)
            loss = combined_loss(pred, mask)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                metrics = compute_seg_metrics(pred, mask)
                ious.append(metrics['iou'])

        losses.append(loss.item())

    return {
        'loss': np.mean(losses),
        'iou': np.mean(ious) if ious else 0,
    }

# ============================================================================
# TRAINING
# ============================================================================

def train_phase(phase_num, epochs, img_size, batch_size, lr, train_p, val_p, test_p, model, history):
    print(f"\n{'='*60}")
    print(f"Phase {phase_num}: epochs={epochs}, img_size={img_size}, lr={lr}, batch={batch_size}")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader, _ = make_loaders(
        train_p, val_p, test_p, img_size, batch_size, batch_eval=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    scaler = GradScaler('cuda')

    best_iou = 0.0
    torch.cuda.empty_cache()

    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, scaler, train=True)
        va = run_epoch(model, val_loader, optimizer, scaler, train=False)
        scheduler.step()

        for k, v in [('train_loss', tr['loss']), ('train_iou', tr['iou']),
                     ('val_loss', va['loss']), ('val_iou', va['iou'])]:
            history[k].append(v)

        if va['iou'] > best_iou:
            best_iou = va['iou']
            torch.save({'model_state': model.state_dict(), 'val_iou': best_iou},
                       CKPT_DIR / 'best.pth')
            improved = ' ✓'
        else:
            improved = ''

        if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(f'P{phase_num} E{epoch:3d}  loss={tr["loss"]:.4f}  val_iou={va["iou"]:.4f}{improved}')

    print(f"Phase {phase_num} done. Best val_iou={best_iou:.4f}\n")
    return best_iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=0,
                        help='Run specific phase (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    # Load dataset
    train_p, val_p, test_p = load_dataset()

    # Build model
    model = build_model().to(DEVICE)
    print(f"Model: MAnet + {ENCODER}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M\n")

    # Load history
    history_path = CKPT_DIR / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"Loaded history ({len(history['train_loss'])} epochs)")
    else:
        history = {k: [] for k in ['train_loss', 'train_iou', 'val_loss', 'val_iou']}

    # Load checkpoint if resuming
    if (args.resume or args.phase > 1) and (CKPT_DIR / 'best.pth').exists():
        ckpt = torch.load(CKPT_DIR / 'best.pth', map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded checkpoint (val_iou={ckpt['val_iou']:.4f})\n")

    # Phase 1
    if args.phase in [0, 1]:
        train_phase(1, 20, 256, 32, 1e-3, train_p, val_p, test_p, model, history)

    # Phase 2
    if args.phase in [0, 2]:
        train_phase(2, 70, 384, 16, 5e-4, train_p, val_p, test_p, model, history)

    # Phase 3
    if args.phase in [0, 3]:
        train_phase(3, 30, 512, 8, 3e-5, train_p, val_p, test_p, model, history)

    # Save history
    with open(CKPT_DIR / 'history.json', 'w') as f:
        json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f)

    # Evaluate on test set
    print("Evaluating on test set...")
    ckpt = torch.load(CKPT_DIR / 'best.pth', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    _, _, test_loader, _ = make_loaders(train_p, val_p, test_p, 512, 4, batch_eval=4)

    test_iou_scores, test_dice_scores = [], []
    with torch.no_grad():
        for img, mask in tqdm(test_loader, desc='Test eval'):
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            pred = model(img)

            pred_binary = (torch.sigmoid(pred) > 0.5).long()
            tp, fp, fn, tn = smp.metrics.get_stats(pred_binary, mask.long(), mode='binary')

            iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item()
            dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro').item()

            test_iou_scores.append(iou)
            test_dice_scores.append(dice)

    test_results = {
        'crack_iou': np.mean(test_iou_scores),
        'dice': np.mean(test_dice_scores),
    }

    print(f"\n{'='*60}")
    print(f"Test Results:")
    print(f"  Crack IoU: {test_results['crack_iou']:.4f}")
    print(f"  Dice     : {test_results['dice']:.4f}")
    print(f"{'='*60}\n")

    # Plot training curves
    if len(history['train_loss']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        ep = range(1, len(history['train_loss']) + 1)
        axes[0].plot(ep, history['train_loss'], label='train')
        axes[0].plot(ep, history['val_loss'], label='val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(ep, history['train_iou'], label='train')
        axes[1].plot(ep, history['val_iou'], label='val')
        axes[1].set_title('IoU')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'training_curves.png', dpi=150)
        print(f"Saved plot: {PLOTS_DIR / 'training_curves.png'}")

    print(f"\nCheckpoint: {CKPT_DIR / 'best.pth'}")
    print(f"History:    {CKPT_DIR / 'history.json'}")

if __name__ == '__main__':
    main()
