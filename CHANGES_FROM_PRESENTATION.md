# Changes from Earlier Presentation to Current State

> Use this as a checklist when updating the report and PowerPoint slides.
> Classification (nb01, nb08) is the only thing that did NOT change.

---

## 1. Detection — biggest change

| | Earlier Presentation | Now |
|---|---|---|
| Model | YOLOv8s | **YOLOv8l** |
| Training data | OmniCrack30k + DACL10K mixed | **OmniCrack30k only** |
| Training approach | Single phase, 640px | **3-phase progressive: 320px → 640px → 1024px** |
| Val mAP@50 | 96.7% | **70.30% (Phase 2)** |
| Test mAP@50 | 34.59% | **55.22% (Phase 2) / 63.01% TTA** |
| Notebook | `02_detection_old.ipynb` (archived) | **`02_detection.ipynb`** |

**Model progression narrative:**

YOLOv8s (original) achieved **96.7% val mAP@50** but only **34.6% test mAP@50** — a large val/test gap caused by domain shift between the training distribution and the held-out test set, compounded by noisy mask→bounding-box annotation conversion from the mixed DACL10K data. Rather than report the inflated val number, we switched to YOLOv8l with 3-phase progressive training on OmniCrack30k-only data, achieving **63.01% TTA mAP@50** on a 581-image test split (notebook 05) — a genuine, test-set-verified improvement.

**Why it changed:** Mixed DACL10K data introduced mask→bounding-box annotation noise that capped detection performance. Removing it and using 3-phase progressive training with a larger backbone (YOLOv8l) resolved it.

**Slides to update:** any slide showing detector mAP, training approach, model architecture.

---

## 2. Segmentation — second biggest change

| | Earlier Presentation | Now |
|---|---|---|
| Model | U-Net + ResNet34 | **MAnet + mit_b2 (SegFormer encoder)** |
| Image size | 384px | **512px** |
| Training | Single phase | **3-phase progressive (256px → 384px → 512px)** |
| Loss function | BCE | **Tversky (0.4) + Lovász (0.4) + Boundary BCE (0.2)** |
| Crack IoU | 68.43% | **96.23%** |
| mIoU (2-class) | 83.56% | **85.67%** |
| Dice | 81.26% | **98.08%** |
| Background IoU | — | **75.11%** |
| Threshold | 0.5 | **0.60 (val IoU sweep)** |
| Notebook | `03_segmentation.ipynb` (v1) | **`03_segmentation.ipynb` (v4 final)** |

**Model progression narrative (all versions in `old_notebooks/`):**

| Version | Notebook | Val IoU | Test IoU | Status |
|---|---|---|---|---|
| v1 — U-Net + ResNet34 | `03_segmentation_retrain_weighted.ipynb` | 89.61% | 83.94% | ✗ below 90% target; masonry data caused FPs |
| v2 — MAnet + mit_b2 (intermediate) | `03_segmentation_inference_fp_old.ipynb` | **97.51%** | — | ✗ sigmoid miscalibrated; 27–32% FP rate on background |
| v3 — MAnet + mit_b2 (no masonry) | `03_segmentation_old_v2.ipynb` | 94.84% | 93.92% | ✗ no post-processing; crackforest failures |
| **v4 — MAnet + mit_b2 (3-phase, final)** | `03_segmentation.ipynb` | 97.62% | **96.23%** | ✓ deployed in webapp |

The peak val IoU of **97.51%** (v2) was achieved but not deployed — the model had a miscalibrated sigmoid output predicting 27–32% of background pixels as crack regardless of threshold. Full retrain with 3-phase progressive training (v4) produced **96.23% test Crack IoU** with properly calibrated predictions and low false positives — this is the production model.

**Extra finding:** Old model had miscalibrated sigmoid output — predicted everything as crack regardless of threshold. Required full retrain. Also replaced broken `webapp/models/segmentor_best.pth`.

**Slides to update:** segmentation architecture diagram, results table, Crack IoU figure, training curves.

---

## 3. Cross-Dataset Evaluation (nb04) — re-run with new models

### Segmentor cross-domain (updated)
| Source | mIoU (before) | mIoU (now) |
|---|---|---|
| Combined in-dist | 83.56% | **85.67%** (from Task 3) |
| CrackForest | 73.61% | **41.87%** |
| Masonry | 86.92% | **51.79%** |

Cross-domain drop is larger with new model — expected, because MAnet+mit_b2 was trained on OmniCrack30k (pavement cracks), not heritage masonry.

### Detector cross-domain (updated)
| Domain | mAP@50 (before) | mAP@50 (now) |
|---|---|---|
| OmniCrack30k in-dist | 34.59% | **63.01% (TTA)** |
| CrackForest | 4.76% | **4.09%** |
| Masonry | 1.13% | **1.74%** |

Cross-domain collapse pattern is the same — just baseline is now much higher.

**Slides to update:** cross-dataset heatmap, radar chart, bar chart, summary table.

---

## 4. Detector Fine-Tuning (nb05) — re-run with new base model

- Base model changed: YOLOv8s → **YOLOv8l** (`detector_v2_retrain_best.pt`)
- Finetuning results (CrackForest + Masonry) are now from YOLOv8l backbone
- Heritage domain adaptation numbers remain directionally the same

**Slides to update:** before/after fine-tuning table if it references the base model name.

---

## 5. Failure Analysis (nb06) — completely different results

### Detection failure mode (new model, different behavior)
| Metric | Before | Now |
|---|---|---|
| True Positives | 152 | **45** |
| False Positives | 628 | **24** |
| False Negatives | **0** | **107** |
| Precision @conf≥0.25 | 0.195 | **0.652** |
| Recall @conf≥0.25 | **1.000** | **0.296** |

**Narrative change:** Old detector fired on all 780 images (recall=1, massive FP). New detector is calibrated but misses heritage cracks (FN=107). Different failure mode: now it's a domain gap miss problem, not a false alarm flood.

### Segmentation failure analysis (cross-domain evaluation)
| Metric | Before | Now |
|---|---|---|
| Crack IoU | 0.6914 | **0.3524** |
| Background IoU | 0.9899 | **0.8375** |
| 2-class mIoU | 0.8407 | **0.5949** |
| Dice (crack) | 0.8064 | **0.4485** |
| % images > 0.80 mIoU | 65.6% | **26.0%** |
| % images < 0.50 crack IoU | 10.6% | **63.4%** |

Note: nb06 evaluates on masonry+crackforest (cross-domain for new model). Lower numbers expected.

### Cross-task consistency
| | Before | Now |
|---|---|---|
| Type A (cls=intact, det fires) | 624 | **24** |
| Type B (cls=cracked, det silent) | 0 | **111** |
| Cls ↔ Det agreement | 20% | **82.7%** |

**Narrative change:** Old detector agreed with classifier only 20% of the time (fired on everything). New detector agrees 82.7% — much better calibrated. Type B = 111 is the new concern.

**Slides to update:** failure analysis slide, confusion matrix, cross-task consistency diagram.

---

## 6. Notebook 09 — DACL10K Multi-Class Detector *(was Notebook 10)*

- **FINAL result: 13.48% overall mAP@50** (YOLOv8s, 50 epochs, 640px) — kept, no retrain
- Retrain attempt (YOLOv8l, 25 epochs) abandoned: GPU quota exhausted at epoch 24/25, best val mAP only 7% — worse than old result

**Per-class final results:**
| Class | mAP@50 |
|---|---|
| crack | 12.71% |
| efflorescence | 9.30% |
| spalling | 16.17% |
| weathering | 22.65% |
| wetspot | 6.57% |
| **all** | **13.48%** |

**Slides to update:** notebook number (10 → 09) if referenced.

---

## 7. SAHI Evaluation — notebook deleted

Old notebook `07_sahi_evaluation.ipynb` removed from repo. Results: plain YOLO 28.0% vs SAHI 18.9% (SAHI underperformed). Notebooks renumbered: old 08→07, old 09→08, old 10→09.

**Slides to update:** any slide referencing notebook numbers 07–10.

---

## 8. Webapp ✓ DONE

| Location | Before | Now |
|---|---|---|
| Homepage stat pill | 55.22% | **63.01%** ✓ |
| Homepage model card | 55.22% | **63.01%** ✓ |
| About page detection badge | mAP@50 55.22% | **mAP@50 63.01%** ✓ |
| About page detection text | "YOLOv8s (small variant)" | **"YOLOv8l (large variant)"** ✓ |
| Before/After example images | none | **synthetic deg pairs added** ✓ |
| segmentor_best.pth | broken (all-crack FP) | **fixed (MAnet+mit_b2, val IoU 0.9762)** ✓ |

---

## 9. What Did NOT Change

- **Classification (nb01)** — EfficientNet-B4, 99.83% acc, 99.56% F1, ≈100% AUC-ROC — untouched
- **Classifier fine-tuning (nb07)** — heritage recall 71.2% → 98.1% — untouched *(renumbered from nb08)*
- **Degradation analysis (nb08)** — Pearson r=0.843, GLCM/Gabor/LoG filter bank — untouched *(renumbered from nb09)*
- **SSIM before/after comparison** — untouched
- **Preprocessing filters** — untouched

---

## 10. Notebook Renaming ✓ DONE

| Old name | New name |
|---|---|
| `02_detection_v2.ipynb` | `02_detection.ipynb` ✓ |
| `07_sahi_evaluation.ipynb` | **deleted** ✓ |
| `08_classifier_finetuning.ipynb` | `07_classifier_finetuning.ipynb` ✓ |
| `09_degradation_analysis.ipynb` | `08_degradation_analysis.ipynb` ✓ |
| `10_dacl10k_detector.ipynb` | `09_dacl10k_detector.ipynb` ✓ |
| Old segmentation notebooks | moved to `old_notebooks/` ✓ |

README, report, webapp About page all updated to match.

---

## 11. Report ✓ DONE (merged + updated)

- Article survey merged into `report.pdf` as Part II — single document submitted
- All metrics updated: detection table (96.7%→70.3% val, 34.6%→55.22% test), abstract, arch description
- YOLOv8l correctly described as large variant (43.6M params), phase training described
- "U-Net/ResNet34" replaced with "MAnet/MiT-B2" in all tables and captions

---

## 12. New: EDA Notebook (`00_eda.ipynb`) ✓ DONE

Added a new exploratory data analysis notebook as the first notebook in the pipeline. All figures saved to `outputs/eda/`.

**Key findings to highlight in the presentation:**

| Finding | Slide relevance |
|---|---|
| 57,291 total images across 6 datasets | Opening dataset slide — replace old totals |
| DACL10K class imbalance: Crack >> Wetspot (~3× more crack annotations) | Motivates focal loss; add to detection methods slide |
| Crack pixel density median 1.31% (CrackForest) — extremely sparse | Justifies IoU over accuracy as segmentation metric |
| OmniCrack30K: 22,158 training pairs, 256×256 bool masks | Update dataset table slide |
| Spatial heatmap: defects uniformly distributed (no positional bias) | Data quality slide |
| RGB histogram shift: DACL10K/CrackForest vs StructDamage | Motivates ImageNet normalisation + domain adaptation narrative |
| t-SNE: CrackForest/HistoricalCrack/Masonry cluster separately | Feature space separation slide / cross-domain explanation |

**Slides to add/update:**
- Add a "Dataset Overview" slide with the horizontal bar chart (`01_dataset_overview.png`)
- Add the DACL10K class distribution + bbox scatter (`04_detection_stats.png`)
- Add the segmentation gallery (`07_segmentation_gallery.png`) — visually strong slide
- Add the spatial heatmap (`05_bbox_heatmap.png`) if slide count allows
- Update any dataset totals (was 27,274 → now 57,291 including OmniCrack30K)

---

## Pending

- [ ] PowerPoint — update slides using tables above (detection, segmentation, cross-dataset, failure analysis, notebook numbers)
- [ ] PowerPoint — add EDA slides (dataset overview bar chart, DACL10K class dist, segmentation gallery)
