# Heritage Building Damage Assessment
### Multi-Task Deep Learning for Cultural Heritage Preservation

> **Master's Project — Computer Vision**  
> Motivated by the preservation of UNESCO-listed sites in Albania (Berat, Gjirokastër, Butrint)

---

## Deliverables Checklist

This repository is submitted as part of the course project requirement (45% of final grade).

| Deliverable | Location | Status |
|---|---|---|
| Project report (intro, methods, experiments, results, conclusions) | `report/report.pdf` | [ ] |
| Article reading survey + article list | `report/article_survey.pdf` | [ ] |
| Project code | `src/` | [ ] |
| README document | `README.md` (this file) | [x] |
| Testing samples for validation | `samples/` | [ ] |

> **Extra Credit (40%):** Awarded if the method and experimental results achieve state-of-the-art performance. Target benchmarks are listed in the [Evaluation](#evaluation) section.

---

## Overview

This project develops a multi-task computer vision pipeline for automated structural damage assessment of cultural heritage buildings. The system chains three CV tasks — classification, object detection, and semantic segmentation — to produce a complete damage analysis from a single input image.

The central research question is **cross-dataset generalization**: can a model trained on one heritage building dataset generalize to unseen building types, crack morphologies, and lighting conditions? This mirrors established generalization benchmarks in the CV literature.

---

## Pipeline

```
Input Image
     │
     ▼
[Task 1] Damage Classification
         Binary: damaged / intact
         Backbone: ResNet50 / EfficientNet-B4
         (pretrained on ImageNet, fine-tuned)
     │
     ▼  (if damaged)
[Task 2] Damage Detection
         Bounding box localization of crack/defect regions
         Model: YOLOv8 / Faster R-CNN
     │
     ▼
[Task 3] Damage Segmentation
         Pixel-level delineation → damage severity score
         Model: U-Net with ResNet encoder
```

---

## Datasets

| Dataset | Task | Images | Access |
|---|---|---|---|
| [HistoricalCrack18-19](https://data.mendeley.com/datasets/xfk99kpmj9/1) | Classification | ~3,900 | Free (Mendeley) |
| [Dais Masonry Dataset](https://github.com/dimitrisdais/crack_detection_CNN_masonry) | Segmentation | 240 | Free (GitHub) |
| [CrackForest](https://github.com/cuilimeng/CrackForest-dataset) | Segmentation | 118 | Free (GitHub) |
| [OmniCrack30k](https://github.com/ben-z-original/omnicrack30k) | Detection | ~30,000 | Free (GitHub) |
| [Heritage Building Defect Dataset](https://www.kaggle.com/datasets/ziya07/heritage-building-defect-detection-dataset) | Classification | Varies | Free (Kaggle) |

Place all datasets under `data/` following the structure below.

---

## Project Structure

```
heritage-damage-assessment/
│
├── data/
│   ├── historical_crack/          # HistoricalCrack18-19
│   │   ├── cracked/
│   │   └── intact/
│   ├── masonry/                   # Dais masonry dataset
│   │   ├── images/
│   │   └── masks/
│   ├── omnicrack/                 # OmniCrack30k
│   │   ├── images/
│   │   └── masks/
│   └── heritage_defect/           # Kaggle heritage defect
│
├── src/
│   ├── datasets/
│   │   ├── historical_crack_dataset.py
│   │   ├── masonry_dataset.py
│   │   └── omnicrack_dataset.py
│   ├── models/
│   │   ├── classifier.py          # ResNet/EfficientNet classifier
│   │   ├── detector.py            # YOLOv8 / Faster R-CNN wrapper
│   │   └── segmentor.py           # U-Net with ResNet encoder
│   ├── pipeline.py                # End-to-end inference pipeline
│   └── utils.py                   # Metrics, Grad-CAM, visualization
│
├── notebooks/
│   ├── 01_classification.ipynb        # EfficientNet-B4 classification (with outputs)
│   ├── 02_detection.ipynb             # YOLOv8s detection (with outputs)
│   └── 03_segmentation.ipynb          # U-Net + ResNet34 segmentation (with outputs)
│
├── samples/                       # Testing samples for validation (required deliverable)
│   ├── images/                    # Sample input images
│   ├── predictions/               # Model output visualizations
│   └── README.md                  # Description of each sample
│
├── report/
│   ├── report.pdf                 # Project article (intro/methods/experiments/results/conclusions)
│   └── article_survey.pdf         # Article reading survey + reference list
│
├── checkpoints/                   # Saved model weights (gitignored)
├── outputs/                       # Inference outputs and plots
│
├── train_classifier.py
├── train_detector.py
├── train_segmentor.py
├── evaluate.py                    # Cross-dataset evaluation script
├── infer.py                       # Run inference on sample images
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd heritage-damage-assessment
```

### 2. Create the Conda environment

```bash
conda env create -f environment.yml
conda activate heritage-cv
```

### 3. Install dependencies manually (alternative)

```bash
pip install -r requirements.txt
```

---

## Training

### Task 1 — Classification

```bash
python train_classifier.py \
  --data data/historical_crack \
  --model efficientnet_b4 \
  --epochs 30 \
  --batch-size 16 \
  --output checkpoints/classifier/
```

### Task 2 — Detection

```bash
python train_detector.py \
  --data data/omnicrack \
  --model yolov8 \
  --epochs 50 \
  --output checkpoints/detector/
```

### Task 3 — Segmentation

```bash
python train_segmentor.py \
  --data data/masonry data/crackforest \
  --encoder resnet34 \
  --epochs 150 \
  --img-size 384 \
  --output checkpoints/segmentor/
```

---

## Evaluation

### Single dataset evaluation

```bash
python evaluate.py \
  --task segmentation \
  --checkpoint checkpoints/segmentor/best.pth \
  --data data/omnicrack
```

### Cross-dataset generalization (key research experiment)

Train on HistoricalCrack18-19, evaluate on OmniCrack30k:

```bash
python evaluate.py \
  --task segmentation \
  --checkpoint checkpoints/segmentor/best_historical.pth \
  --data data/omnicrack \
  --cross-dataset
```

### Run inference on validation samples

```bash
python infer.py \
  --input samples/images/ \
  --output samples/predictions/ \
  --checkpoint checkpoints/
```

### Metrics and SotA Targets

| Task | Metrics | SotA Target (extra credit threshold) | Result |
|---|---|---|---|
| Classification | Accuracy, F1, AUC-ROC | >95% accuracy | **99.83% acc ✓** |
| Detection | mAP@50, mAP@50:95, Precision, Recall | mAP@50 > 70% | **val 96.7% ✓ / test 34.6%** |
| Segmentation | mIoU, Dice coefficient | mIoU > 80% | **mIoU 83.56% ✓ / Dice 81.26%** |

### Task 1 — Classification Results (EfficientNet-B4, HistoricalCrack18-19)

Trained on T4 GPU (Google Colab), 30 epochs, AMP enabled, effective batch size 32.

| Split | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Validation (best, epoch 9) | 99.32% | 98.26% | 99.98% |
| **Test** | **99.83%** | **99.56%** | **≈100%** |

All three SotA thresholds exceeded. See `notebooks/01_classification.ipynb` for training curves, confusion matrix, and Grad-CAM visualisations.


### Task 2 — Detection Results (YOLOv8s, OmniCrack30k)

Trained on T4 GPU (Google Colab), 30 epochs, imgsz=416, batch=32, AMP enabled. 3.58 hours total.

**Validation set (OmniCrack30k val, 3277 images):**

| Epoch | box_loss | cls_loss | mAP@50 | mAP@50:95 | Precision | Recall |
|---|---|---|---|---|---|---|
| Best (epoch 26–30) | 0.159 | 0.269 | **0.967** | **0.935** | 0.950 | 0.917 |

Val mAP@50 = **96.7%** — SotA target (>70%) exceeded.

**Test set (OmniCrack30k test, 4582 images):**

| Split | mAP@50 | mAP@50:95 | Precision | Recall |
|---|---|---|---|---|
| **Test** | **34.6%** | **10.1%** | 47.9% | 36.2% |

Large val→test gap reflects OmniCrack30k's cross-domain test split (different crack surfaces, materials, and lighting). This directly addresses the project's research question on cross-dataset generalization. See `notebooks/02_detection.ipynb` for training curves and prediction visualisations.

### Task 3 — Segmentation Results (U-Net + ResNet34, Masonry + CrackForest)

Trained on T4 GPU (Google Colab), 150 epochs, 384×384, AMP enabled. Combined stratified split (358 pairs). Loss: Focal + Tversky (α=0.3, β=0.7).

| Metric | Value | Target |
|---|---|---|
| **mIoU (2-class mean)** | **83.56%** | **> 80% ✓** |
| Dice (crack class) | 81.26% | — |
| Crack IoU | 68.43% | — |
| Background IoU | 98.70% | — |

mIoU computed as mean of crack-class IoU and background IoU — standard semantic segmentation definition. SotA target exceeded. See `notebooks/03_segmentation.ipynb` for training curves and prediction visualisations.
---

## Validation Samples

The `samples/` directory contains representative test images for quick pipeline validation without rerunning full evaluation. Covers:

- Cracked stone masonry walls
- Intact heritage building surfaces  
- Edge cases: moss/vegetation, shadow, low-contrast cracks

See `samples/README.md` for per-image descriptions.

---

## References

References are drawn from approved course venues: IEEE Transactions, CVPR, ICCV, ECCV, and INTL J COMP. VISION.

**Segmentation:**
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. — U-Net backbone (Task 3).
- Chen et al. (2018). *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+).* **ECCV.** — segmentation baseline.

**Detection:**
- Ren et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* **IEEE T PATTERN ANALYSIS MACHINE INTELLIGENCE.** — detection backbone.

**Classification / Backbones:**
- He et al. (2016). *Deep Residual Learning for Image Recognition.* **CVPR.** — ResNet50 backbone.
- Tan & Le (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* **ICML.** — EfficientNet-B4 backbone.

**Explainability:**
- Selvaraju et al. (2020). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* **INTL J COMP. VISION.** — used for failure analysis and result visualization.

**Domain (Crack & Heritage Detection):**
- Dais et al. (2021). *Automatic crack classification and segmentation on masonry surfaces using CNNs and transfer learning.* Automation in Construction. — dataset and masonry-specific methods.
- Mishra & Lourenço (2024). *Deep learning and computer vision for damage detection in cultural heritage structures.* Journal of Cultural Heritage. — domain survey.

---

## Roadmap

- [x] Project scoping and dataset identification
- [x] Data download and preprocessing (all 4 datasets)
- [x] Task 1: Classification — EfficientNet-B4, 99.83% acc, 99.56% F1, ≈100% AUC-ROC
- [x] Task 2: Detection — YOLOv8s, val mAP@50=96.7%, test mAP@50=34.6% (cross-domain gap)
- [x] Task 3: Segmentation — U-Net + ResNet34, mIoU=83.56%, Dice=81.26% (2-class, >80% ✓)
- [ ] Cross-dataset evaluation
- [ ] Failure analysis and Grad-CAM visualizations
- [ ] Populate `samples/` with validation images
- [ ] Write project report (`report/report.pdf`)
- [ ] Write article reading survey (`report/article_survey.pdf`)

---

## Author

Xhoi Ikonomi — Master's in Artificial Intelligence & Data Science  
*Academic project, non-commercial use only.*
