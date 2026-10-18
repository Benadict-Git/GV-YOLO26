# GV-YOLO26: Project Work Summary & Development Log

> **Project Title**: GV-YOLO26: Lightweight Real-Time Detection of Speed Bumps and Potholes Toward Intelligent Vehicle Suspension Systems  
> **Course / Degree**: U23ADS705 – Mini Project, B.Tech Artificial Intelligence and Data Science  
> **Institution**: Department of Information Technology, Sona College of Technology (Autonomous), Anna University, Chennai  
> **Authors**: Benadict Infant A (`61782323110022`), Elangovan S (`61782323110035`)  
> **Supervisor**: Dr. P. Shanmugaraja (Assistant Professor, Department of Information Technology)  
> **Head of Department**: Dr. J. Akilandeswari (Professor & Head, Department of Information Technology)  
> **Date**: October 2026  

---

## 1. Executive Summary

This project develops **GV-YOLO26**, an ultra-lightweight, end-to-end, Non-Maximum Suppression (NMS)-free neural network designed to detect roadway surface hazards—specifically **speed bumps** and **potholes**—from monocular camera feeds for active vehicle suspension preview control.

Built on the latest **YOLO26n** architecture, GV-YOLO26 redesigns the feature backbone and neck using cheap convolutions (**GhostConv**, **VoVGSCSPC**), incorporates an extra high-resolution **P2 detection branch** (160 × 160 grid) for small and distant hazards, mitigates head complexity via a **Lightweight Grouped Detection Head (LGDH)**, and trains with a **Normalized Wasserstein Distance (NWD)** box regression loss alongside synthetic weather augmentations.

All experiments, ablation studies, baseline benchmarks, and video pipelines were trained and evaluated on the **SBP-YOLO dataset** ([speed12](https://www.kaggle.com/datasets/elangovan001/speed12) on Kaggle). The final project deliverable includes complete source code, Kaggle training workflows, trained weights, evaluation logs, and an audited 78-page academic project report in both [DOCX](file:///home/benny/Projects/GV-YOLO26/GV_YOLO26_Project_Report.docx) and [PDF](file:///home/benny/Projects/GV-YOLO26/GV_YOLO26_Project_Report.pdf) formats.

---

## 2. Architectural Design & Technical Innovations

| Component | Technical Implementation | Purpose & Design Rationale |
| :--- | :--- | :--- |
| **Base Model** | [`YOLO26n`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/models/gv-yolo26.yaml) | Provides end-to-end NMS-free inference via dual-branch one-to-one matching; eliminates Distribution Focal Loss (DFL) post-processing. |
| **P2 Detection Level** | Stride-4 feature branch (160 × 160 resolution) | Captures small and distant potholes (where 46.5% of potholes are under 32 pixels in size) before spatial downsampling loses feature detail. |
| **LGDH Head** | [`class LGDH`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/lgdh.py) (Grouped Convolutions, `channels_per_group=8`) | Offsets the high FLOP cost of the P2 feature map by grouping stem convolutions, keeping head parameter count low. |
| **GhostConv** | [`class GhostConv`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/ghost.py) | Replaces expensive standard strided downsampling in deeper stages (P4/16 and P5/32) with intrinsic linear transformations. |
| **VoVGSCSPC** | [`class VoVGSCSPC`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/gsconv.py) (GSConv + Channel Shuffle) | Replaces heavy C3k2 feature blocks in the neck and deep backbone to preserve spatial feature mixing with reduced FLOP overhead. |
| **NWD Box Loss** | $\text{NWD}(A, B) = \exp\left(-\frac{\sqrt{W_2^2(A, B)}}{C}\right)$, $\alpha=0.5$ | Mitigates IoU sensitivity on tiny bounding boxes where minor pixel shifts produce zero IoU overlap. |
| **Weather Augmentation** | [`weather_augmentations()`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/augment.py) (Albumentations) | Simulates adverse driving conditions: synthetic rain, snow, motion blur, and illumination shifts. |
| **Temporal Video Tracking** | [`demo_video.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/demo_video.py) with ByteTrack ($k=3, \tau=0.35$) | Associates bounding boxes across frames; suppresses transient single-frame false detections before warning the driver. |

---

## 3. Incremental 6-Stage Ablation Ladder (Verified Results)

To verify each architectural modification independently, a rigorous 6-stage ablation ladder was designed, trained for 150 epochs each on Kaggle Tesla T4 GPUs, and evaluated on a clean, leakage-free held-out set:

| Stage | Configuration | Parameters | GFLOPs (640) | Checkpoint | mAP@0.5 | mAP@0.5:0.95 | PyTorch FP32 FPS (T4) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **YOLO26n Baseline** | 2.38 M | 5.3 | 5.2 MB | 81.0% | 48.6% | 58.0 FPS (17.25 ms) |
| **2** | **+ P2 Head** | 2.40 M | 6.6 | 5.3 MB | 80.3% | 47.6% | 55.7 FPS (17.97 ms) |
| **3** | **+ LGDH Head** | 2.37 M | 6.2 | 5.2 MB | 80.0% | 46.5% | 57.4 FPS (17.44 ms) |
| **4** | **+ GhostConv** | 2.08 M | 5.8 | 4.6 MB | 81.5% | 47.2% | 56.1 FPS (17.84 ms) |
| **5** | **+ VoVGSCSPC** | 1.57 M | 5.1 | 3.7 MB | 77.9% | 45.1% | 59.2 FPS (16.90 ms) |
| **6** | **+ NWD Loss (GV-YOLO26)** | **1.57 M** | **5.1** | **3.7 MB** | **77.5%** | **45.8%** | **57.5 FPS (17.38 ms)** |

### Key Ablation Insights:
- **Efficiency Gains**: GV-YOLO26 achieves a **33.7% parameter reduction** (1.57M vs 2.38M) and a **3.6% FLOP reduction** (5.1 vs 5.3 GFLOPs), while supporting an extra 160 × 160 detection level.
- **Latency Characteristics**: Single-image batch=1 latency remains virtually constant (~57.5 to 58.0 FPS) because GPU kernel launch overhead and intermediate tensor memory traffic for the high-resolution P2 grid balance out arithmetic FLOP reductions.
- **Per-Class Dynamics**: Speed bumps are predominantly wide targets (median width: 72.0% of image width), benefiting primarily from deeper semantic stride levels; potholes benefit from P2 high resolution, but reduced capacity in cheaper layers creates a trade-off.

---

## 4. Dataset Engineering & Data Leakage Resolution

All models were developed on the **SBP-YOLO Dataset** ([speed12](https://www.kaggle.com/datasets/elangovan001/speed12) on Kaggle by `elangovan001` / Liang et al.):

### Dataset Partition Structure (Table 4.4):
- **Total Images**: 7,600 annotated images (12,205 potholes, 3,217 speed bumps).
- **Original Folders**: `train/` (5,725 images), `valid/` (750 images), `test/` (1,125 images).
- **Carved Validation Split (`val_carved`)**: 586 images (seeded 10% sample of `train/`, seed 42) for checkpoint selection and early stopping.
- **Active Training Split (`train_used`)**: 5,243 images (8,300 potholes, 2,144 speed bumps).

### Data Leakage Discovery & Clean Held-Out Set:
1. **Audit Finding**: Filename hash analysis revealed that 526 raw test images and 378 valid images were noise/crop-augmented duplicates of training photos (sharing visual hashes like `1386_jpg.rf.df5cd630...`). Furthermore, sequential GoPro/dashcam drives (`G00xxxxx.jpg`) overlapped across splits.
2. **Resolution via [`tools/make_clean_split.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/tools/make_clean_split.py)**: Filtered out all images sharing source photos or video drives with `train/`.
3. **Leakage-Free Held-Out Set**: Yielded **482 clean evaluation images** (286 from test, 196 from valid; 354 potholes, 384 speed bumps) ensuring zero train-test overlap.
4. **Impact on Metrics**: Raw test split evaluation showed inflated metrics by 2 to 3 percentage points (e.g., 82.1% raw test mAP50 vs 77.5% clean held-out mAP50 for GV-YOLO26), proving the academic necessity of the held-out set.

---

## 5. Knowledge Distillation Experiments & Honest Findings

Knowledge distillation was explored using an S-scale teacher model (**GV-YOLO26-S**: 6.11M parameters, 19.2 GFLOPs):
- **Distillation Loss Formulation**: Combined Bridging Cross-task Knowledge Distillation (BCKD) on detection outputs with Channel-Wise Distillation (CWD) on intermediate neck features.
- **Experimental Runs**:
  - *From scratch (150 epochs)*: Diverged at epoch 3 during warm-up; loss became undefined (`NaN`).
  - *Fine-tuning stage 6 (50 epochs)*: Degraded validation mAP from 81.3% down to 49.4%; distillation loss turned negative (reaching -1260).
- **Root-Cause Diagnosis**: YOLO26 decodes box distances without DFL or coordinate positivity constraints. Roughly 0.9% of student raw predictions decoded to negative widths/heights, causing Generalized IoU (GIoU) between teacher and student to range from -10 to +1 rather than $[-1, 1]$. Consequently, the term $1 - \text{GIoU}$ was unbounded below, driving gradients to minus infinity.
- **Academic Reporting**: Reported honestly as a negative result with the proposed fix (bounding coordinate decoding and valid-anchor masking) slated for future work.

---

## 6. Real-World Video Tracking & Hazard Warning Demonstration

A real-time video detection and warning pipeline was built in [`demo_video.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/demo_video.py):
- **Algorithm**: ByteTrack multi-object tracking coupled with a 3-frame temporal confirmation filter:
  $$\text{show}(t) \iff \text{frames}(t) \ge 3 \quad \text{and} \quad \text{conf}(t) \ge 0.35$$
- **Benchmark Across 4 Challenging Test Clips**:
  - `v1.mp4` *(Daytime, rain)*: 240 frames, 98 hazard frames, filtered 8 raw tracks $\to$ 6 confirmed hazards.
  - `v2.mp4` *(Night)*: 240 frames, 111 hazard frames, filtered 12 raw tracks $\to$ 6 confirmed hazards.
  - `v3.mp4` *(Night)*: 240 frames, 134 hazard frames, filtered 17 raw tracks $\to$ 12 confirmed hazards.
  - `v4.mp4` *(Dusk)*: 240 frames, 39 hazard frames, filtered 7 raw tracks $\to$ 2 confirmed hazards.
- **Results**: Reduced 44 raw transient detections down to 26 confirmed hazards, effectively eliminating flickering false alarms. Ran at **~5.0 FPS end-to-end on a standard laptop CPU** (Intel Core i5-1240P).

---

## 7. Project Report Engineering, Audit, and Finalization

An automated, reproducible report generation suite was engineered in [`report_build/`](file:///home/benny/Projects/GV-YOLO26/report_build/) to ensure text and numbers never diverge:

```
report_build/
├── prepare_inputs.py             # Consolidates benchmark JSONs, dataset stats, and figures
├── narrative.py                  # Generates all results-dependent prose bound to metrics
├── docxlib.py                    # Low-level XML parser, run merger, and table editor
├── build_report.py               # Assembles the college report document
├── fix_pages.py                  # Multi-pass LibreOffice PDF renderer & TOC/LOT/LOF page synchronizer
├── GV_YOLO26_Project_Report.docx # Final generated Word document (4.0 MB)
├── GV_YOLO26_Project_Report.pdf  # Final compiled PDF document (2.9 MB, 78 pages)
└── backup/                       # Pre-audit backup archive
```

### Key Report Corrections Applied:
1. **Unified Dataset**: Removed all references to a phantom Roboflow v8 dataset. Refactored Chapter 4, Chapter 5, and Table 4.4 to describe solely the single SBP-YOLO dataset.
2. **Table Cleanliness**: Completely eliminated all blank cells, hyphens (`—`), and `TBD` placeholders across all tables (Tables 4.1, 4.2, 4.4, 4.6, 5.1–5.7, 6.1).
3. **Framing of TensorRT & Deployment**: Removed completed TensorRT claims from Chapter 5 main results. Evaluated completed inference speed in PyTorch FP32 on Tesla T4 (57.5 FPS). Moved TensorRT INT8/FP16 compilation, Quantization-Aware Training (QAT), and NVIDIA Jetson AGX Xavier deployment into **Section 6.2 (Future Scope)**.
4. **Ablation Validation**: Confirmed and thoroughly documented the 6-stage ablation ladder.
5. **Page Number Stabilization**: Successfully synchronized all 89 page references in Table of Contents, List of Tables, and List of Figures with exact physical pages via headless LibreOffice rendering passes.

---

## 8. Repository Structure & Artifact Map

```
GV-YOLO26/
├── GV_YOLO26_Project_Report.docx     # Primary editable college mini-project report
├── GV_YOLO26_Project_Report.pdf      # Print-ready compiled 78-page PDF report
├── summary.md                        # This project work log
│
├── GV-YOLO26/                        # Main project code tree
│   ├── models/                       # Model architecture specifications
│   │   ├── gv-yolo26.yaml            # Final GV-YOLO26 network definition (N and S scales)
│   │   └── ablation/                 # Incremental ablation YAMLs (s2_p2, s3_lgdh, s4_ghost)
│   ├── gv_modules/                   # Custom PyTorch modules & trainers
│   │   ├── ghost.py                  # GhostConv implementation
│   │   ├── gsconv.py                 # GSConv and VoVGSCSPC blocks
│   │   ├── lgdh.py                   # Lightweight Grouped Detection Head
│   │   ├── nwd.py                    # Normalized Wasserstein Distance loss
│   │   ├── augment.py                # Albumentations synthetic weather transforms
│   │   └── distill.py                # BCKD and CWD distillation loss functions
│   ├── tools/
│   │   ├── benchmark.py              # Automated accuracy, cost, and latency evaluation
│   │   ├── make_clean_split.py       # Data leakage detection & clean held-out set builder
│   │   ├── model_summary.py          # Parameter and GFLOP profiling utility
│   │   └── make_figures.py           # Report plot and figure generators
│   ├── train.py                      # Training entrypoint supporting all ablation stages
│   ├── distill.py                    # Knowledge distillation trainer
│   ├── val.py                        # Validation & mAP evaluation script
│   ├── infer.py                      # Image inference utility
│   ├── demo_video.py                 # Dashcam video detection with ByteTrack temporal confirmation
│   ├── export_trt.py                 # TensorRT engine export utility (for future deployment)
│   ├── Kaggle_Train_GV_YOLO26.ipynb  # End-to-end self-contained Kaggle execution notebook
│   ├── benchmark/train_baselines.py  # 7-detector stock baseline training script
│   ├── data/data.yaml                # SBP-YOLO dataset configuration
│   └── weights_kaggle/               # Trained Kaggle model checkpoints (s1 through s6 + teacher)
│
└── report_build/                     # Report generation and page stabilization pipeline
```

---

## 9. Next Steps & Future Scope

1. **Quantization-Aware Training (QAT)**:
   - Preliminary post-training INT8 quantization dropped mAP@0.5:0.95 to 35.0%. Future work will integrate PyTorch QAT during the final 20 fine-tuning epochs to preserve 8-bit precision on edge accelerators.
2. **Physical Embedded Deployment**:
   - Flash and benchmark GV-YOLO26 TensorRT FP16/INT8 engines on dedicated vehicular boards (**NVIDIA Jetson AGX Xavier** or **Jetson Orin Nano**).
3. **Active Suspension Controller Interfacing**:
   - Interface detections with automotive CAN bus protocols to feed preview damping signals to active semi-active magnetorheological or electromagnetic dampers.
4. **Localized Indian Road Hazard Dataset**:
   - Collect and label high-resolution datasets covering unpainted speed breakers, rural road crests, and waterlogged monsoon potholes.

