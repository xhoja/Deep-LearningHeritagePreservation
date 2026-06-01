"""3-Phase Segmentation Retrain for Hyperstack — MAnet/mit_b2 Progressive Resizing

SETUP:
  1. Create GPU VM on Hyperstack (A100/A6000, 20GB+ RAM)
  2. SSH into VM
  3. Upload data: scp -r data/masonry user@ip:/root/data/
  4. Upload data: scp -r data/crackforest user@ip:/root/data/
  5. Copy this script: scp retrain_seg_3phase_hyperstack.py user@ip:/root/
  6. SSH in and run: python retrain_seg_3phase_hyperstack.py
  7. After training: scp user@ip:/root/checkpoints/segmentor_v4/best.pth .
  8. Copy checkpoint to webapp/models/segmentor_best.pth
"""
import shutil
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import cv2
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

print(f"Device: {DEVICE}")

# ============================================================================
# DATASET
# ============================================================================
class CrackSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.images = sorted([f for f in self.img_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png']])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = (img_t - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
        mask_t = torch.from_numpy(mask).float() / 255.0

        return img_t, mask_t.unsqueeze(0)

# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc='Train', leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0
    all_iou = []

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc='Val', leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            masks_bin = (masks > 0.5).long()

            tp = ((preds == 1) & (masks_bin == 1)).sum()
            fp = ((preds == 1) & (masks_bin == 0)).sum()
            fn = ((preds == 0) & (masks_bin == 1)).sum()
            iou = tp / (tp + fp + fn + 1e-7)

            total_loss += loss.item()
            all_iou.append(iou.item())

    return total_loss / len(loader), np.mean(all_iou)

def phase_train(model, phase, img_size, epochs, lr, train_loader, val_loader, device, ckpt_dir):
    """Train one phase"""
    print(f"\n{'='*70}")
    print(f"Phase {phase}: {img_size}px, {epochs} epochs, lr={lr}")
    print(f"{'='*70}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_iou = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss, val_iou = val_epoch(model, val_loader, device)

        print(f"[P{phase} {epoch+1:2d}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}", end="")

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_iou': val_iou,
                'phase': phase,
                'encoder': 'mit_b2',
                'img_size': img_size,
            }
            torch.save(ckpt, ckpt_dir / 'best.pth')
            print(f"  ✓")
        else:
            print()

    print(f"Phase {phase} done. Best val_iou={best_iou:.4f}")
    return best_iou

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Check data exists
    print("\n=== Checking data ===")
    data_paths = {
        'masonry_img': Path('data/masonry/images'),
        'masonry_mask': Path('data/masonry/masks'),
        'crackforest_img': Path('data/crackforest/images'),
        'crackforest_mask': Path('data/crackforest/masks'),
    }

    for name, path in data_paths.items():
        if not path.exists():
            print(f"ERROR: {path} not found!")
            print("Upload data first:")
            print("  scp -r data/masonry user@ip:/root/data/")
            print("  scp -r data/crackforest user@ip:/root/data/")
            exit(1)
        count = len(list(path.glob('*')))
        print(f"  ✓ {name}: {count} files")

    ckpt_dir = Path('checkpoints/segmentor_v4')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Build model once
    print("\n=== Building model ===")
    model = smp.MAnet(
        encoder_name='mit_b2',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(DEVICE)

    # ========================================================================
    # PHASE 1: 256px, 20 epochs
    # ========================================================================
    print("\n=== Phase 1: 256px ===")
    train_ds_p1 = CrackSegDataset('data/masonry/images', 'data/masonry/masks', 256)
    val_ds_p1 = CrackSegDataset('data/crackforest/images', 'data/crackforest/masks', 256)
    train_loader_p1 = DataLoader(train_ds_p1, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_p1 = DataLoader(val_ds_p1, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds_p1)} | Val: {len(val_ds_p1)}")
    best_p1 = phase_train(model, 1, 256, 20, 0.001, train_loader_p1, val_loader_p1, DEVICE, ckpt_dir)

    # ========================================================================
    # PHASE 2: 384px, 70 epochs
    # ========================================================================
    print("\n=== Phase 2: 384px ===")
    train_ds_p2 = CrackSegDataset('data/masonry/images', 'data/masonry/masks', 384)
    val_ds_p2 = CrackSegDataset('data/crackforest/images', 'data/crackforest/masks', 384)
    train_loader_p2 = DataLoader(train_ds_p2, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_p2 = DataLoader(val_ds_p2, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds_p2)} | Val: {len(val_ds_p2)}")
    best_p2 = phase_train(model, 2, 384, 70, 0.0001, train_loader_p2, val_loader_p2, DEVICE, ckpt_dir)

    # ========================================================================
    # PHASE 3: 512px, 60 epochs
    # ========================================================================
    print("\n=== Phase 3: 512px ===")
    train_ds_p3 = CrackSegDataset('data/masonry/images', 'data/masonry/masks', 512)
    val_ds_p3 = CrackSegDataset('data/crackforest/images', 'data/crackforest/masks', 512)
    train_loader_p3 = DataLoader(train_ds_p3, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader_p3 = DataLoader(val_ds_p3, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_ds_p3)} | Val: {len(val_ds_p3)}")
    best_p3 = phase_train(model, 3, 512, 60, 0.00005, train_loader_p3, val_loader_p3, DEVICE, ckpt_dir)

    # ========================================================================
    # DONE
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Phase 1 (256px): {best_p1:.4f}")
    print(f"Phase 2 (384px): {best_p2:.4f}")
    print(f"Phase 3 (512px): {best_p3:.4f}")
    print(f"\nCheckpoint: {ckpt_dir / 'best.pth'}")
    print(f"\nDownload checkpoint:")
    print(f"  scp user@ip:/root/checkpoints/segmentor_v4/best.pth .")
    print(f"  cp best.pth webapp/models/segmentor_best.pth")

if __name__ == '__main__':
    main()
