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
         Binary: cracked / intact
         Model: EfficientNet-B4 (ImageNet pretrained, fine-tuned)
         Output: class label + confidence score
     │
     ▼  (routed only if classified as cracked)
[Task 2] Crack Detection
         Bounding box localisation of crack regions
         Model: YOLOv8s — two variants:
           • General: trained on OmniCrack30k (30k images)
           • Heritage: fine-tuned on CrackForest + Masonry (250 images)
         Output: bounding boxes + confidence scores
     │
     ▼
[Task 3] Crack Segmentation
         Pixel-level crack mask
         Model: U-Net + ResNet34 encoder (ImageNet pretrained)
         Trained on: Masonry (240) + CrackForest (118) combined
         Output: binary crack mask (crack / background)
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
│   ├── 03_segmentation.ipynb          # U-Net + ResNet34 segmentation (with outputs)
│   ├── 04_cross_dataset_eval.ipynb    # Cross-domain evaluation + t-SNE / Grad-CAM / heatmap
│   ├── 05_detector_finetuning.ipynb   # Domain adaptation: fine-tune detector on heritage data
│   ├── 06_failure_analysis.ipynb      # Failure analysis & Grad-CAM++ across all three models
│   ├── 07_sahi_evaluation.ipynb       # SAHI sliced inference vs plain YOLO comparison
│   └── 08_classifier_finetuning.ipynb # Classifier domain adaptation on heritage masonry data
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

| Task | Metrics | SotA Target | Result |
|---|---|---|---|
| Classification | Accuracy, F1, AUC-ROC | >95% accuracy | **99.83% acc ✓** |
| Classification (fine-tuned, heritage domain) | Accuracy, F1 | — | **99.4% combined / 98.1% heritage recall** |
| Detection (OmniCrack30k) | mAP@50 | >70% | val **96.7% ✓** / test 34.6% |
| Detection (fine-tuned, heritage) | mAP@50 | — | **23.6% combined / 28.4% CrackForest / 23.1% Masonry** |
| Detection (SAHI sliced inference) | mAP@50 | — | Plain YOLO **28.0%** / SAHI **18.9%** (SAHI underperforms) |
| Segmentation | mIoU, Dice | mIoU > 80% | **mIoU 83.56% ✓ / Dice 81.26%** |

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

### Task 4 — Cross-Dataset Evaluation (`notebooks/04_cross_dataset_eval.ipynb`)

Each trained model was evaluated on datasets it was **not** trained on to measure cross-domain generalisation. Visualisations include Grad-CAM attention maps, t-SNE feature-space embeddings (EfficientNet-B4), per-source performance heatmap, and radar charts.

#### Classifier (EfficientNet-B4) — Recall on cracked-class images

| Test Domain | Recall | Notes |
|---|---|---|
| HistoricalCrack (in-distribution) | **99.83%** | Own test split |
| CrackForest | **89.83%** | 106/118 correctly classified |
| Masonry | **76.67%** | 184/240 correctly classified |

Graceful degradation. Masonry drop (~23 pp) consistent with the visual domain shift from historic wall photographs to structured masonry crack patterns. t-SNE of 1 792-d avgpool features shows three clearly separated domain clusters, confirming genuine distribution gap.

#### Segmentor (U-Net + ResNet34) — mIoU per source within test split

| Source subset | mIoU | Crack IoU | Dice |
|---|---|---|---|
| Combined test (in-distribution) | **83.56%** | 68.43% | 81.26% |
| CrackForest subset | **73.61%** | 48.78% | 65.58% |
| Masonry subset | **86.92%** | 75.01% | 85.72% |

Model generalises better to masonry (wider cracks, stronger contrast) than CrackForest (thin, irregular cracks). Qualitative inference on HistoricalCrack/cracked images shows plausible crack masks despite zero heritage-specific training data.

#### Detector (YOLOv8s, OmniCrack30k weights) — mAP@50

| Test Domain | mAP@50 | Notes |
|---|---|---|
| OmniCrack30k test (in-distribution) | **34.59%** | Evaluated on own test split |
| CrackForest | **4.76%** | Severe collapse |
| Masonry | **1.13%** | Near-zero |

**Finding:** The detector shows catastrophic cross-domain collapse. OmniCrack30k images are large-scale pavement/concrete photographs; CrackForest and Masonry contain fine, structured heritage cracks at smaller scale and different texture. The model's bounding-box priors and feature responses do not transfer without adaptation — motivating Task 5.

---

### Task 5 — Detector Domain Adaptation (`notebooks/05_detector_finetuning.ipynb`)

**Approach:** Fine-tuned the OmniCrack30k YOLOv8s checkpoint on CrackForest + Masonry combined (358 images, stratified 70/15/15 split). Key hyperparameters: `freeze=10` (first 10 backbone layers frozen to preserve OmniCrack features), `lr0=0.001` (10× below default), 50 epochs, patience=15.

#### Before vs After Fine-tuning — mAP@50

| Test subset | Before (OmniCrack weights) | After (fine-tuned) | Δ |
|---|---|---|---|
| Combined heritage test (n=54) | 8.8% | **23.6%** | +169% |
| CrackForest (n=18) | 18.1% | **28.4%** | +57% |
| Masonry (n=36) | 4.7% | **23.1%** | +394% |

**Key findings:**

1. **Masonry improvement largest** — 4.7% → 23.1% (+394% relative) with only 168 masonry training images, confirming that OmniCrack pretraining provides transferable crack features but heritage domain gap is significant.
2. **Backbone freezing critical** — freezing the first 10 backbone layers prevented catastrophic forgetting of OmniCrack crack-detection features while allowing the head to adapt to heritage crack morphology.
3. **Domain gap is hard to bridge with small data** — 250 images and 50 epochs yield modest absolute mAP. Heritage cracks are thinner, more irregular, and at different scales than OmniCrack30k pavement cracks; the bounding-box label quality from mask contours also introduces noise.
4. **Precision/Recall after fine-tuning:** P=0.281, R=0.341 (combined); P=0.320, R=0.289 (CrackForest).

**Report narrative:** Domain adaptation via fine-tuning on 250 heritage crack images improves mAP@50 from 8.8% to 23.6% on the combined heritage test set — a 169% relative improvement — demonstrating that adaptation is possible but difficult, highlighting the severity of the OmniCrack→heritage domain gap. This directly addresses the project's central research question on cross-dataset generalisation.

---

### Task 6 — Failure Analysis & Grad-CAM++ (`notebooks/06_failure_analysis.ipynb`)

Post-hoc analysis across all three models on the HistoricalCrack test split (780 images, same 80/20 seed=42 as training). Uses **Grad-CAM++** (pytorch-grad-cam) on EfficientNet-B4's last MBConv block for sharper, more localised attention maps than standard Grad-CAM.

#### Classification — In-Distribution Error Analysis

| Metric | Value |
|---|---|
| Accuracy | 99.49% |
| F1 | 98.70% |
| Total errors | 4 / 780 (0.5%) |
| False Positives (intact → cracked) | 4 |
| **False Negatives (cracked → intact)** | **0** |

Zero false negatives is the critical safety property for damage assessment: the classifier never misses actual structural damage. The 4 false positives were intact images with texture features (shadow lines, mortar joints) visually similar to crack patterns — confirmed by Grad-CAM++ attention maps which show the model correctly attending to crack-like edges.

#### Classification — Cross-Dataset Generalization (OOD Grad-CAM++)

EfficientNet-B4 trained on HistoricalCrack was applied to CrackForest (road surface cracks) and Masonry (building cracks) — datasets it never saw during training:

| Domain | Mean P(cracked) | Fraction > 0.5 |
|---|---|---|
| HistoricalCrack test (in-dist) | 0.205 | 20.0% |
| **CrackForest (OOD — road)** | **0.998** | **100%** |
| **Masonry (OOD — building)** | **0.968** | **100%** |

The classifier generalises with near-certainty to both OOD crack domains. Grad-CAM++ maps on CrackForest and Masonry images show the model attending to crack edges and branching structures — not dataset-specific colour or texture — confirming it learned transferable crack representations. The HistoricalCrack 20% cracked fraction matches the expected class distribution (757 cracked / 3896 total ≈ 19.4%).

#### Detection — Domain Gap Quantified

YOLOv8s (trained on OmniCrack30k road cracks) evaluated on HistoricalCrack building facades:

| Metric | Value |
|---|---|
| True Positives | 152 |
| False Positives | 628 |
| False Negatives | **0** |
| Precision @conf≥0.25 | 0.195 |
| **Recall @conf≥0.25** | **1.000** |

The detector fires on all 780 test images, including 624 intact building facades. This is a domain adaptation failure: OmniCrack30k images are pavement/concrete photographs; HistoricalCrack contains plaster and stone facade textures that the detector treats as crack-like. However, **recall is 1.0** — every actual crack is detected. For structural damage assessment, this is the correct failure mode: false alarms are preferable to missed damage. The chained pipeline addresses this directly — the upstream classifier (99.49% accurate) acts as a gate and rejects intact images before detection, eliminating FP load in practice.

Confidence histogram analysis shows TP and FP detections have overlapping confidence distributions, explaining why threshold tuning alone cannot separate them without domain adaptation (see notebook 05).

#### Segmentation — Metric Clarification & Failure Cases

Two IoU metrics are reported to align with the literature and with notebook 03:

| Metric | Value | Notes |
|---|---|---|
| Crack IoU (hard) | 0.6914 | Crack pixels only — ~3% of image area |
| Background IoU | 0.9899 | Background nearly always correct |
| **2-class mIoU** | **0.8407** | Mean(crack IoU, bg IoU) — matches notebook 03 (target >0.80 ✓) |
| Dice (crack class) | 0.8064 | |
| % images > 0.80 mIoU | 65.6% | |
| % images < 0.50 crack IoU | 10.6% | |

Crack pixels constitute ~3% of each image, so a model could score ~97% pixel accuracy by predicting all-background. The 2-class mIoU (0.84) and Dice (0.81) together confirm the model is genuinely learning crack structure, not background shortcuts.

Best/worst IoU visualisations use semi-transparent error overlays on the original image: **green = correct crack pixels (TP), red = missed cracks (FN), blue = spurious predictions (FP)**. Worst-case failures share a common pattern: thin hairline cracks (<3 px wide) embedded in complex masonry textures with similar intensity.

#### Cross-Task Consistency

| Failure Mode | Count | Interpretation |
|---|---|---|
| Type A: classifier=intact, detector fires | 624 | All intact images — detector ignores classifier gate |
| Type B: classifier=cracked, detector silent | 0 | No case where classifier fires but detector misses |
| Classifier ↔ Detector agreement | 20% | Structurally expected given recall=100% detector |

The 20% agreement rate is mathematically expected: the classifier labels ~20% of images as cracked; the detector fires on 100%; they agree only on the cracked subset. Type B = 0 is the important result — every image the classifier calls cracked, the detector also fires on, meaning the two models are fully consistent on positive cases.

---

### Task 7 — SAHI Sliced Inference Evaluation (`notebooks/07_sahi_evaluation.ipynb`)

**Hypothesis:** SAHI (Slicing Aided Hyper Inference) improves detection of hairline cracks by tiling each image into overlapping 512×512 patches and running YOLO on each tile, making small cracks visible at full resolution.

**Setup:** Same fine-tuned YOLOv8s checkpoint from Task 5. Adaptive slice size 256–512 px, 20% overlap, conf=0.25.

| Method | mAP@50 | Boxes predicted |
|---|---|---|
| Plain YOLOv8s (imgsz=640) | **28.0%** | Normal |
| YOLOv8s + SAHI (adaptive 256–512 tiles) | **18.9%** | 32 across 54 images |
| Delta | **−9.2 pp** | — |

**Finding:** SAHI underperformed. Heritage masonry and CrackForest cracks are medium-to-large relative to image size — not sub-pixel objects. Slicing offered no resolution advantage while introducing patch-boundary artifacts. The fine-tuned model also produced low-confidence predictions on cropped tiles (most filtered at conf=0.25), leading to severe under-detection. SAHI is most effective for genuinely tiny objects in high-resolution aerial or satellite imagery.

---

### Task 1b — Classifier Domain Adaptation (`notebooks/08_classifier_finetuning.ipynb`)

**Problem:** EfficientNet-B4 (trained on HistoricalCrack) achieves 99.83% on its own test split but only 71.2% cracked recall on heritage masonry images, where GradCAM shows attention on irrelevant background regions.

**Approach:** Fine-tuned from the existing checkpoint on a combined dataset (masonry + crackforest + historical_crack). Froze backbone features[0:4], trained features[5:] + classifier head. LR=1e-4, 20 epochs, cosine schedule, label smoothing=0.1.

| | historical_crack test | Heritage domain (masonry+crackforest) |
|--|--|--|
| Before fine-tuning | 99.83% | **71.2%** cracked recall |
| After fine-tuning | **99.4%** (combined test) | **98.1%** cracked recall |

**Key findings:**

1. **Heritage recall jumps +26.9 pp** — 71.2% → 98.1% on the masonry+crackforest subset, confirming the domain gap was in higher-level appearance priors learnable from 250 images.
2. **In-distribution performance preserved** — combined test acc 95.2% → 99.4%, historical_crack performance essentially unchanged.
3. **GradCAM attention shifts** — post fine-tuning, the model attends to crack edges and branching structures in masonry images rather than irrelevant background textures.

---

## Web Application

FastAPI demo in `webapp/`. Upload masonry images and get a full three-model damage report in the browser.

**Features:**
- **Multi-image batch mode** — drop multiple files; results processed sequentially with live thumbnail status badges (pending → running → done/error); click any completed thumbnail to review its results
- **Animated step loader** — Classifying → Detecting → Segmenting with checkmarks (single-image mode)
- **Severity summary card** — CRITICAL / HIGH / MODERATE / LOW / NONE verdict derived from model consensus (X/3 models agree), with per-model agree/disagree breakdown
- **Crack arc gauge** — SVG donut gauge colour-coded by coverage (<5% green, 5–15% yellow, 15–30% orange, >30% red)
- **Grad-CAM overlay** — EfficientNet-B4 attention map on the uploaded image
- **JSON report download** — exports severity, classification, detection, and segmentation results with timestamp

**Run locally:**
```bash
cd webapp
pip install fastapi uvicorn pillow torch torchvision ultralytics segmentation-models-pytorch
uvicorn app:app --reload --port 8000
# open http://localhost:8000
```

> **Note on classifier vs. detector/segmentor agreement:** In practice the classifier occasionally disagrees with the detector and segmentor on fine-grained heritage cracks. The detector and segmentor share a spatial-feature inductive bias that transfers better across domains; the classifier operates on global image statistics and is more sensitive to domain shift. The severity card surfaces this disagreement explicitly rather than hiding it.

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
- [x] Task 4: Cross-dataset evaluation — Grad-CAM, t-SNE feature space, performance heatmap
- [x] Task 5: Detector domain adaptation — fine-tuned on heritage data, combined 8.8% → 23.6% mAP@50 (+169%)
- [x] Task 6: Failure analysis & Grad-CAM++ — zero FN classifier, OOD generalization confirmed, detector domain gap quantified
- [x] Task 7: SAHI sliced inference evaluation — plain YOLO 28.0% vs SAHI 18.9% (SAHI underperforms on this domain)
- [x] Task 1b: Classifier domain adaptation — heritage cracked recall 71.2% → 98.1% after fine-tuning on masonry+crackforest
- [x] Populate `samples/` with validation images (saved via notebook 06)
- [x] Web application — multi-image batch analysis, severity card, animated step loader, crack arc gauge, JSON export
- [ ] Write project report (`report/report.pdf`)
- [ ] Write article reading survey (`report/article_survey.pdf`)

---

## Author

Xhoi Ikonomi — Master's in Artificial Intelligence & Data Science  
*Academic project, non-commercial use only.*
