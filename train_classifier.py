"""
Task 1 — Classification training script.

Usage:
    python train_classifier.py \
        --data data/historical_crack \
        --model efficientnet_b4 \
        --epochs 30 \
        --batch-size 32 \
        --output checkpoints/classifier/
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).parent))
from src.datasets.historical_crack_dataset import HistoricalCrackDataset
from src.models.classifier import build_classifier, INPUT_SIZES
from src.utils import compute_metrics, plot_training_curves, save_history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_sampler(dataset: HistoricalCrackDataset) -> WeightedRandomSampler:
    sample_weights = [dataset.class_weights[label] for label in dataset.labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                 replacement=True)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    desc = "train" if train else "val"
    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, desc=desc, leave=False, ncols=80):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = total_loss / len(all_labels)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       default='data/historical_crack')
    p.add_argument('--model',      default='efficientnet_b4',
                   choices=['resnet50', 'efficientnet_b4'])
    p.add_argument('--epochs',     type=int,   default=30)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--output',     default='checkpoints/classifier/')
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--workers',    type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_size = INPUT_SIZES[args.model]
    print(f"Model: {args.model}  |  Input size: {img_size}x{img_size}")

    # Datasets
    train_ds = HistoricalCrackDataset(args.data, split='train', img_size=img_size,
                                       random_state=args.seed)
    val_ds   = HistoricalCrackDataset(args.data, split='val',   img_size=img_size,
                                       random_state=args.seed)
    test_ds  = HistoricalCrackDataset(args.data, split='test',  img_size=img_size,
                                       random_state=args.seed)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"Class weights (intact/cracked): {train_ds.class_weights}")

    pin = device.type == "cuda"
    sampler = make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.workers,
                              pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=pin)

    # Model
    model = build_classifier(args.model, num_classes=2, pretrained=True)
    model = model.to(device)

    # Loss with class weights to reinforce imbalance handling
    class_weights = torch.tensor(train_ds.class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ['train_loss', 'train_acc', 'val_loss',
                                'val_acc', 'val_f1', 'val_auc_roc']}
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        history['train_loss'].append(train_m['loss'])
        history['train_acc'].append(train_m['accuracy'])
        history['val_loss'].append(val_m['loss'])
        history['val_acc'].append(val_m['accuracy'])
        history['val_f1'].append(val_m['f1'])
        history['val_auc_roc'].append(val_m['auc_roc'])

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"loss {train_m['loss']:.4f} acc {train_m['accuracy']:.4f} | "
              f"val_loss {val_m['loss']:.4f} val_acc {val_m['accuracy']:.4f} "
              f"val_f1 {val_m['f1']:.4f} val_auc {val_m['auc_roc']:.4f}")

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_f1': best_f1, 'val_acc': val_m['accuracy'],
                        'args': vars(args)},
                       output_dir / 'best.pth')
            print(f"  -> Saved best model (val_f1={best_f1:.4f})")

    # Final checkpoint
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict(),
                'args': vars(args)},
               output_dir / 'last.pth')

    # Test evaluation with best model
    ckpt = torch.load(output_dir / 'best.pth', map_location=device)
    model.load_state_dict(ckpt['model_state'])
    test_m = run_epoch(model, test_loader, criterion, None, device, train=False)
    print(f"\n=== Test Results ===")
    print(f"Accuracy : {test_m['accuracy']:.4f}")
    print(f"F1       : {test_m['f1']:.4f}")
    print(f"AUC-ROC  : {test_m['auc_roc']:.4f}")

    history['test'] = test_m
    save_history(history, output_dir)
    plot_training_curves(history, output_dir)
    print(f"\nCheckpoints and plots saved to {output_dir}")


if __name__ == '__main__':
    main()
