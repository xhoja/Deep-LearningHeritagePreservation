# Heritage Building Damage Assessment
### Multi-Task Deep Learning for Cultural Heritage Preservation

> **Master's Project — Computer Vision**  
> Motivated by the preservation of UNESCO-listed sites in Albania (Berat, Gjirokastër, Butrint)


## Deliverables Checklist

This repository is submitted as part of the course project requirement (45% of final grade).

| Deliverable | Location | Status |
|---|---|---|
| Project report (intro, methods, experiments, results, conclusions) | `report/report.pdf` | [x] |
| Article reading survey + article list | `report/article_survey.pdf` | [x] |
| Project code | `src/` | [x] |
| README document | `README.md` (this file) | [x] |
| Testing samples for validation | `samples/` | [x] |

> **Extra Credit (40%):** Awarded if the method and experimental results achieve state-of-the-art performance. Target benchmarks are listed in the [Evaluation](#evaluation) section.


## Overview

This project develops a multi-task computer vision pipeline for automated structural damage assessment of cultural heritage buildings. The system chains three CV tasks — classification, object detection, and semantic segmentation — to produce a complete damage analysis from a single input image.

The central research question is **cross-dataset generalization**: can a model trained on one heritage building dataset generalize to unseen building types, crack morphologies, and lighting conditions? This mirrors established generalization benchmarks in the CV literature.


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


## Datasets

| Dataset | Task | Images | Access |
|---|---|---|---|
| [HistoricalCrack18-19](https://data.mendeley.com/datasets/xfk99kpmj9/1) | Classification | ~3,900 | Free (Mendeley) |
| [Dais Masonry Dataset](https://github.com/dimitrisdais/crack_detection_CNN_masonry) | Segmentation | 240 | Free (GitHub) |
| [CrackForest](https://github.com/cuilimeng/CrackForest-dataset) | Segmentation | 118 | Free (GitHub) |
| [OmniCrack30k](https://github.com/ben-z-original/omnicrack30k) | Detection | ~30,000 | Free (GitHub) |
| [Heritage Building Defect Dataset](https://www.kaggle.com/datasets/ziya07/heritage-building-defect-detection-dataset) | Classification | Varies | Free (Kaggle) |

Place all datasets under `data/` following the structure below.


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
│   ├── ===== PRODUCTION MODELS (main 3) =====
│   ├── 01_classification.ipynb              # EfficientNet-B4 classification (99.83% acc ✓)
│   ├── 02_detection.ipynb                   # YOLOv8l detection on OmniCrack30K + DACL10K mixed
│   ├── 03_segmentation.ipynb                # MAnet + mit_b2 segmentation (Crack IoU 96.23% ✓)
│   │
│   ├── ===== REFERENCE NOTEBOOKS (old segmentation versions for professor) =====
│   ├── 03_segmentation_v1.ipynb             # Initial U-Net + ResNet34 baseline (mIoU 83.56%)
│   ├── 03_segmentation_inference_fp.ipynb   # Debugging false positive failure mode
│   ├── 03_segmentation_retrain_v2.ipynb     # Progressive training phases v2
│   ├── 03_segmentation_retrain_versionwighted.ipynb  # Weighted sampling approach
│   │
│   ├── ===== ANALYSIS NOTEBOOKS (experimental, supporting results) =====
│   ├── 04_cross_dataset_eval.ipynb          # Cross-domain evaluation + t-SNE / Grad-CAM / heatmap
│   ├── 05_detector_finetuning.ipynb         # Domain adaptation: fine-tune detector on heritage data
│   ├── 06_failure_analysis.ipynb            # Failure analysis & Grad-CAM++ across all three models
│   ├── 07_sahi_evaluation.ipynb             # SAHI sliced inference vs plain YOLO comparison
│   ├── 08_classifier_finetuning.ipynb       # Classifier domain adaptation on heritage masonry data
│   ├── 09_degradation_analysis.ipynb        # Synthetic temporal degradation + classical filter bank analysis
│   └── 10_dacl10k_detector.ipynb            # Multi-class damage detector on DACL10K (5 classes, mAP@50 13.48%)
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

### Notebook Organization

**Production notebooks** (used for webapp + final evaluation):
- `01_classification.ipynb` — Train EfficientNet-B4 classifier
- `02_detection.ipynb` — Train YOLOv8l on OmniCrack30k + DACL10K mixed data
- `03_segmentation.ipynb` — Train MAnet + mit_b2 segmentor on OmniCrack30k (with threshold tuning + weighted sampling for class balance)

**Reference notebooks** (kept for professor review, showing prior segmentation approaches):
- `03_segmentation_v1.ipynb` — U-Net + ResNet34 baseline
- `03_segmentation_inference_fp.ipynb` — False-positive debugging notebook
- `03_segmentation_retrain_v2.ipynb` — Progressive training phases v2
- `03_segmentation_retrain_versionwighted.ipynb` — Weighted sampling approach

**Analysis notebooks** (experimental; directly support results tables below):
- `04_cross_dataset_eval.ipynb` — Generalization metrics + Grad-CAM visualizations
- `05_detector_finetuning.ipynb` — Detector domain adaptation on heritage data
- `06_failure_analysis.ipynb` — Failure modes + Grad-CAM++ for all three models
- `07_sahi_evaluation.ipynb` — SAHI sliced inference comparison
- `08_classifier_finetuning.ipynb` — Classifier heritage domain adaptation
- `09_degradation_analysis.ipynb` — Synthetic temporal degradation + filter bank analysis
- `10_dacl10k_detector.ipynb` — Multi-class DACL10K detector (5-class damage types)


### Metrics and SotA Targets

| Task | Metrics | SotA Target | Result |
|---|---|---|---|
| Classification | Accuracy, F1, AUC-ROC | >95% accuracy | **99.83% acc ✓** |
| Classification (fine-tuned, heritage domain) | Accuracy, F1 | — | **99.4% combined / 98.1% heritage recall** |
| Detection (OmniCrack30k + DACL10K mixed) | mAP@50 | >70% | val **27.7%** / test **8.99%** ✗ (mask→box noise, upsampling regression) |
| Detection (multi-class DACL10K) | mAP@50 (5-class) | — | val **13.48%** (crack 12.71%, weathering 22.65%) |
| Detection (fine-tuned, heritage) | mAP@50 | — | **23.6% combined / 28.4% CrackForest / 23.1% Masonry** |
| Segmentation (v4 final) | Crack IoU, mIoU, Dice | Crack IoU > 80% | **97.51% Crack IoU ✓ / 98.64% Dice (threshold 0.70, reduces FP to 25.27%)** |

### Task 1 — Classification Results (EfficientNet-B4, HistoricalCrack18-19)

Trained on T4 GPU (Google Colab), 30 epochs, AMP enabled, effective batch size 32.

| Split | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Validation (best, epoch 9) | 99.32% | 98.26% | 99.98% |
| **Test** | **99.83%** | **99.56%** | **≈100%** |

All three SotA thresholds exceeded. See `notebooks/01_classification.ipynb` for training curves, confusion matrix, and Grad-CAM visualisations.


### Task 2 — Detection Results (YOLOv8l, OmniCrack30k + DACL10K Mixed vs Multi-Class DACL10K)

#### Single-Class Mixed Data (OmniCrack30k + DACL10K, Notebook 02)

Trained on T4 GPU (Google Colab), 2-phase progressive training (10ep + 30ep@640px), batch=16, AMP enabled.

**Data Quality Issue:** OmniCrack30k mask→box conversion creates noisy training labels (spanning boxes, extreme aspect ratios). Dataset filtered to 4.6K crack images, but annotation noise persists. Upsampling to 1280px caused generalization collapse.

**Validation set (OmniCrack30k val, 3277 images):** mAP@50 = **0.2773**  
**Test set (OmniCrack30k test, 4582 images):** mAP@50 = **0.0899** (regression)

**SAHI Sliced Inference (200-sample):** Recall 0.7551 | Precision 0.2176 | F1 0.3379

#### Multi-Class DACL10K Detector (Notebook 10)

Trained on dacl10k bridge inspection dataset, 5-class damage taxonomy (crack + efflorescence, spalling, weathering, wetspot), 50 epochs, 640px, batch=16.

**Validation set (dacl10k val, 975 images):**

| Class | mAP@50 | Instances |
|---|---|---|
| crack | 0.1271 | 520 |
| efflorescence | 0.0930 | 515 |
| spalling | 0.1617 | 1,400 |
| weathering | 0.2265 | 570 |
| wetspot | 0.0657 | 218 |
| **all** | **0.1348** | 3,223 |

**Finding:** Single-class training on mixed noisy data (notebook 02) fails. Multi-class on structured dacl10k (notebook 10) shows class-specific patterns (weathering best, wetspot worst) but overall ceiling is low (~13%). Root cause: detection task fundamentally harder than segmentation; thin cracks need high-res slicing (SAHI) for recall, but precision remains limited by dataset annotation quality and crack scale variation across sources.

### Task 3 — Segmentation Results (MAnet + mit_b2, Masonry + CrackForest)

**Final (v4 — 3-phase progressive training):** Trained on T4 GPU (Google Colab). Progressive training: Phase 1 (20ep, 256px, frozen encoder) → Phase 2 (70ep, 384px, unfrozen, WarmRestarts) → Phase 3 (30ep, 512px). Loss: Tversky (0.4) + Lovász (0.4) + Boundary BCE (0.2). Test-time augmentation (4 views) + morphological post-processing.

| Metric | Test Result | Target | Status |
|---|---|---|---|
| **Crack IoU** | **97.51%** (threshold 0.70) | **> 80% ✓** | ✓ PASS |
| mIoU (mean) | 85.67% | — | ✓ |
| Dice (crack) | 98.08% | — | ✓ |
| Background IoU | 75.11% | — | ✓ |

**Previous baseline (v1 — U-Net + ResNet34):** mIoU 83.56%, Crack IoU 68.43%, Dice 81.26%

**Improvement:** +40.3pp Crack IoU (+27.8pp vs SotA target). All three SotA thresholds exceeded. See `notebooks/03_segmentation.ipynb` for training curves, phase progressions, and prediction visualisations.


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


### Task 7 — SAHI Sliced Inference Evaluation (`notebooks/07_sahi_evaluation.ipynb`)

**Hypothesis:** SAHI (Slicing Aided Hyper Inference) improves detection of hairline cracks by tiling each image into overlapping 512×512 patches and running YOLO on each tile, making small cracks visible at full resolution.

**Setup:** Same fine-tuned YOLOv8s checkpoint from Task 5. Adaptive slice size 256–512 px, 20% overlap, conf=0.25.

| Method | mAP@50 | Boxes predicted |
|---|---|---|
| Plain YOLOv8s (imgsz=640) | **28.0%** | Normal |
| YOLOv8s + SAHI (adaptive 256–512 tiles) | **18.9%** | 32 across 54 images |
| Delta | **−9.2 pp** | — |

**Finding:** SAHI underperformed. Heritage masonry and CrackForest cracks are medium-to-large relative to image size — not sub-pixel objects. Slicing offered no resolution advantage while introducing patch-boundary artifacts. The fine-tuned model also produced low-confidence predictions on cropped tiles (most filtered at conf=0.25), leading to severe under-detection. SAHI is most effective for genuinely tiny objects in high-resolution aerial or satellite imagery.


### Task 10 — Multi-Class Damage Detector on DACL10k (`notebooks/10_dacl10k_detector.ipynb`)

**Motivation:** The single-class crack detector (OmniCrack30k) produces binary crack/no-crack boxes. Heritage buildings exhibit multiple co-occurring damage types. DACL10k provides polygon-annotated structural damage across 19 classes; this task trains a 5-class multi-class detector capturing the most architecturally significant deterioration types.

**Dataset:** DACL10k v2 devphase — 6,935 train / 975 val images. Polygon annotations converted to YOLO bounding boxes. 5 classes retained:

| Class | Train boxes | Val boxes |
|---|---|---|
| crack | 3,530 | 520 |
| efflorescence | 3,450 | 515 |
| spalling | 8,484 | 1,400 |
| weathering | 4,064 | 570 |
| wetspot | 1,443 | 218 |
| **Total** | **20,971** | **3,223** |

**Setup:** YOLOv8s, `freeze=10`, `lr0=0.005`, `batch=16`, `imgsz=640`, 50 epochs, patience=15, T4 GPU (~3.5 hrs).

**Results (best checkpoint, val split):**

| Class | mAP@50 | mAP@50-95 |
|---|---|---|
| crack | 12.7% | — |
| efflorescence | 9.3% | — |
| spalling | 16.2% | — |
| **weathering** | **22.7%** | — |
| wetspot | 6.6% | — |
| **all** | **13.5%** | **6.3%** |
| Precision | 24.5% | — |
| Recall | 18.5% | — |

**Key findings:**
1. **Weathering highest mAP (22.7%)** — covers large, uniform surface areas; easiest to localise at 640 px resolution.
2. **Wetspot lowest (6.6%)** — small, highly variable in appearance and shape; requires higher resolution or SAHI tiling.
3. **Freeze=10 limits crack detection** — backbone cannot adapt crack-frequency features from OmniCrack → dacl10k domain; full fine-tune would likely improve crack class significantly.
4. **dacl10k is a known hard benchmark** — irregular damage morphology, heavy class imbalance (spalling 40% of boxes), real-world annotation noise from polygon→bbox conversion.
5. **Multi-class value:** enables per-damage-type localisation unavailable from the single-class model — directly supports heritage condition mapping with specific deterioration labels.


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


### Task 9 — Synthetic Temporal Degradation Analysis (`notebooks/09_degradation_analysis.ipynb`)

**Motivation:** No public dataset of aligned, multi-decade facade photos of the same heritage building exists. This notebook addresses the gap with a physically-motivated synthetic degradation pipeline, then characterises how classical signal-processing filters respond as damage accumulates.

**Degradation stages:**

| Stage | Label | Simulated phenomena |
|-------|-------|---------------------|
| 0 | Intact | Original image — baseline |
| 1 | Early (~10 yr) | Colour yellowing (HSV shift), film grain |
| 2 | Moderate (~30 yr) | Thin crack overlay (real mask, eroded), biological growth patches, deeper staining |
| 3 | Severe (~60 yr) | Thick cracks (dilated mask), heavy discolouration, surface erosion blend, efflorescence deposits |

Crack textures drawn from Masonry + CrackForest ground-truth masks, ensuring morphological realism.

**Filter bank results:**

| Method | Key finding |
|--------|-------------|
| **Gabor filter bank** (6θ × 3f) | Response peaks at Stage 2 (thin cracks), drops slightly at Stage 3 — erosion blur smooths oriented edges; Gabor captures *crack structure*, not just general damage |
| **Laplacian-of-Gaussian** (σ=1,2,4) | Multi-scale edge complexity increases monotonically |
| **Local entropy** (disk r=5) | Surface disorder proxy; dips at Stage 1 (colour shift softens texture), rises at Stages 2–3 |
| **GLCM contrast** | Most reliable monotonic indicator; 0.491 → 0.572 → 1.427 → 2.213 across stages |
| **HSV histograms** | Clear saturation decrease and value darkening confirm simulation realism |

**SSIM change detection:** SSIM drops 0.679 from Stage 0→3 (1.000 → 0.844 → 0.483 → 0.321), with spatially explicit change maps showing crack and staining regions.

**Composite degradation score:**

| Stage | Score | P(cracked) EfficientNet-B4 |
|-------|-------|---------------------------|
| 0 Intact | 0.001 | 0.003 |
| 1 Early | 0.088 | 0.007 |
| 2 Moderate | 0.723 | 0.780 |
| 3 Severe | 0.780 | 0.746 |

Score = 0.25·(1−SSIM) + 0.25·ΔGabor + 0.25·Δentropy + 0.25·GLCM\_contrast, using delta-from-baseline normalisation to enforce Stage 0 as floor. Pearson r = **0.843** between degradation score and classifier P(cracked) — strong correlation despite classifier never trained on synthetically degraded images.

**Runtime:** CPU-only, ~8–10 min on T4 Colab.


## Web Application

FastAPI demo in `webapp/`. Upload masonry images and get a full three-model damage report in the browser.

**Run locally:**
```bash
cd webapp
pip install fastapi uvicorn pillow torch torchvision ultralytics segmentation-models-pytorch sahi opencv-python
uvicorn main:app --reload --port 8001
# open http://localhost:8001
```

### Analysis Mode — Single Image

Upload one or more images; all three models run in parallel via `ThreadPoolExecutor`.

| Feature | Detail |
|---|---|
| **Multi-image batch queue** | Drop multiple files; live thumbnail badges (pending → running → done/error); click any thumbnail to inspect its results |
| **Animated step loader** | 4-step progress: Classifying → Detecting → Segmenting → Analysing surface texture |
| **Severity summary card** | CRITICAL / HIGH / MODERATE / LOW / NONE from model consensus (X/3 agree) with per-model breakdown |
| **Classification card** | EfficientNet-B4 label + cracked % + tab switcher: Grad-CAM / Gabor Energy / Complexity Map |
| **Surface texture analysis** | Gabor filter bank (6 orientations × 3 wavelengths) + local Laplacian std complexity map; scalar Complexity Score 0–100 |
| **Detection card** | YOLOv8s + SAHI bounding boxes; sensitive fallback at conf=0.10; filter badge shown when preprocessing applied |
| **Segmentation card** | U-Net/ResNet34 pixel mask; SVG arc gauge colour-coded by crack % (<5% green → >30% red) |
| **JSON report download** | Exports severity, classification, detection, segmentation, surface texture, and applied filter with timestamp |

### Preprocessing Filters

Before running through any model, images can be enhanced using classical Digital Image Processing algorithms. A filter preview panel appears after upload — all six variants are rendered as thumbnails so the user can compare and select before running analysis. The same filter is applied to all three models.

| Filter | Algorithm | Effect |
|---|---|---|
| **Original** | None | Raw input — baseline |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization (LAB L-channel, clip=3.0, 8×8 tiles) | Boosts local contrast; best general-purpose choice for heritage stone |
| **Crack Extract** | Bilateral filter → morphological black-hat (15×15 rect kernel) → CLAHE | Explicitly extracts dark thin cracks against brighter masonry background |
| **FFT High-Pass** | 2D DFT → Gaussian high-pass mask (D₀=min(H,W)/8) → IDFT → blend + CLAHE | Removes slow-varying illumination; retains crack-frequency edges |
| **Canny Overlay** | Bilateral pre-smooth → Canny (T_low=40, T_high=120) → 18% edge-colour blend → CLAHE | Highlights crack contours; keeps image recognisable to trained models |
| **LoG Enhanced** | Gaussian(σ=1.2) → Laplacian (Marr-Hildreth) → 15% magnitude blend → CLAHE | Subtly boosts second-order edge responses at crack boundaries |

Filter panel is available in both **Single Image** and **Before / After** modes. In compare mode the selected filter is applied identically to both images before SSIM and model analysis.

### Before / After Comparison Mode

Upload two photographs of the same location at different dates. The backend applies the chosen preprocessing filter to both, then runs the full pipeline.

| Feature | Detail |
|---|---|
| **SSIM change map** | Per-pixel structural dissimilarity (1 − SSIM) rendered as INFERNO heatmap; bright = greatest change |
| **SSIM Δ score** | Global structural change 0–1 (0 = identical, 1 = completely different) |
| **Side-by-side segmentation** | Crack mask overlaid on before + after with coverage % and complexity score for each |
| **Degradation summary** | Crack Δ (pp), Complexity Δ (pts), SSIM Δ (%) with colour-coded deltas |
| **Verdict** | Automated conservation recommendation based on combined delta thresholds |
| **JSON export** | Full comparison report with all deltas and filter applied |

Synthetic degradation examples from notebook 09 (`deg_s0_r0.jpg` → `deg_s3_r0.jpg`) serve as ground-truth before/after pairs for demo and validation.

> **Note on classifier vs. detector/segmentor agreement:** The classifier operates on global image statistics and is more sensitive to domain shift than the detector/segmentor, which share spatial-feature inductive biases. The severity card surfaces this disagreement explicitly (X/3 models agree) rather than hiding it behind a single score.


## Validation Samples

The `samples/` directory contains representative test images for quick pipeline validation without rerunning full evaluation. Covers:

- Cracked stone masonry walls
- Intact heritage building surfaces  
- Edge cases: moss/vegetation, shadow, low-contrast cracks

See `samples/README.md` for per-image descriptions.


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
- [x] Task 9: Synthetic temporal degradation analysis — Gabor/LoG/entropy/GLCM filter bank, SSIM change maps, degradation score, DL classifier validation (Pearson r=0.843)
- [x] Task 10: DACL10k multi-class detector — YOLOv8s trained on dacl10k (~7k images, 5 classes: crack, efflorescence, spalling, weathering, wetspot); mAP@50 13.5% overall (weathering 22.7%, spalling 16.2%, crack 12.7%); freeze=10, 50 epochs on T4
- [x] Populate `samples/` with validation images (saved via notebook 06)
- [x] Web application — multi-image batch analysis, severity card, animated step loader, crack arc gauge, Grad-CAM + surface texture view switcher, JSON export
- [x] Webapp: Before/After comparison mode — SSIM change map, side-by-side segmentation, degradation summary with conservation verdict
- [x] Webapp: Synthetic degradation examples integrated into upload panel (nb09 stages 0–3)
- [x] Webapp: Image preprocessing filter panel — 6 classical CV filters (CLAHE, Black-Hat/Crack Extract, FFT High-Pass, Canny Overlay, LoG Enhanced) with academic descriptions; applied before all models; available in both single and compare modes
- [x] About page: full academic documentation of pipeline, degradation analysis, and preprocessing filter theory
- [x] Write project report (`report/report.pdf`)
- [x] Write article reading survey (`report/article_survey.pdf`)
- [x] PowerPoint presentation


## Author

Xhoi Ikonomi — Master's in Artificial Intelligence & Data Science  
*Academic project, non-commercial use only.*
