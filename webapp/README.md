---
title: Albanian Heritage Damage Assessment
emoji: 🏛
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Albanian Heritage Damage Assessment

AI-powered structural damage detection for Albanian UNESCO World Heritage sites
(Berat, Gjirokastër, Butrint).

**Three-task pipeline:**
- Classification — EfficientNet-B4 (99.8% accuracy)
- Detection — YOLOv8s (mAP@50 = 96.7%) [YOLOv8l v2 in progress]
- Segmentation — MAnet/mit_b2 (Crack IoU = 96.23%, mIoU = 85.67%, Dice = 98.08%) [3-phase progressive training]

## Model weights

Place trained model weights in `models/`:

```
models/
  classifier_best.pth   ← checkpoints/classifier/best.pth from training
  segmentor_best.pth    ← checkpoints/segmentor_v4/best.pth from phase 3 (val_iou 0.9762)
  detector_best.pt      ← YOLOv8s best.pt from training run
```

Upload via `git lfs` (weights are tracked with LFS).
