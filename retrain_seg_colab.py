"""Quick segmentation retrain for Colab — MAnet/mit_b2 on Masonry + CrackForest"""
import json
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

# ============================================================================
# SETUP
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT_DIR = Path('checkpoints/segmentor_v4')
CKPT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

print(f"Device: {DEVICE}")
print(f"Checkpoint dir: {CKPT_DIR}")

# ============================================================================
# DATASET
# ============================================================================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=512):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.images = sorted([f for f in self.img_dir.glob('*') if f.suffix in ['.jpg', '.png']])
        self.transform_img = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.mask_dir / img_path.name

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
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

# ============================================================================
# MAIN
# ============================================================================
def main():
    # Mount drive
    print("\n=== Mounting Google Drive ===")
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = Path('/content/drive/MyDrive/HeritagePreservation')

    # Copy data
    print("\n=== Setting up data ===")
    import subprocess
    subprocess.run(['mkdir', '-p', 'data'], check=False)
    subprocess.run(['cp', '-r', str(DRIVE_DIR / 'data' / 'masonry'), 'data/'], check=False)
    subprocess.run(['cp', '-r', str(DRIVE_DIR / 'data' / 'crackforest'), 'data/'], check=False)

    # Load datasets
    print("Loading datasets...")
    train_ds = SegDataset('data/masonry/images', 'data/masonry/masks', 512)
    val_ds = SegDataset('data/crackforest/images', 'data/crackforest/masks', 512)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_ds)} images")
    print(f"Val: {len(val_ds)} images")

    # Model
    print("\n=== Building model ===")
    model = smp.MAnet(
        encoder_name='mit_b2',
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    print("\n=== Training ===")
    best_iou = 0
    for epoch in range(40):
        train_loss = train_epoch(model, train_loader, opt, DEVICE)
        val_loss, val_iou = val_epoch(model, val_loader, DEVICE)

        print(f"[{epoch+1:2d}/40] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} val_iou={val_iou:.4f}", end="")

        if val_iou > best_iou:
            best_iou = val_iou
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_iou': val_iou,
                'phase': 3,
                'encoder': 'mit_b2',
                'img_size': 512,
            }
            torch.save(ckpt, CKPT_DIR / 'best.pth')
            print(f"  ← SAVED")
        else:
            print()

    print(f"\n=== Done ===")
    print(f"Best val_iou: {best_iou:.4f}")
    print(f"Saved: {CKPT_DIR / 'best.pth'}")

    # Copy to drive
    print("\nSaving to Drive...")
    drive_ckpt_dir = DRIVE_DIR / 'segmentor_best_retrain'
    drive_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(CKPT_DIR / 'best.pth', drive_ckpt_dir / 'best.pth')
    print(f"✓ Checkpoint saved to {drive_ckpt_dir / 'best.pth'}")
    print(f"\nDownload from Drive and copy to webapp/models/segmentor_best.pth")

if __name__ == '__main__':
    main()
