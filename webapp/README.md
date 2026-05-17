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
- Detection — YOLOv8s (mAP@50 = 96.7%)
- Segmentation — U-Net/ResNet34 (mIoU = 83.6%)

## Model weights

Place trained model weights in `models/`:

```
models/
  classifier_best.pth   ← checkpoints/classifier/best.pth from training
  segmentor_best.pth    ← checkpoints/segmentor/best.pth from training
  detector_best.pt      ← YOLOv8s best.pt from training run
```

Upload via `git lfs` (weights are tracked with LFS).
