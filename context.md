# GV-YOLO26: Comprehensive Project Context & Technical Dossier

> **Document Purpose**: Complete context and technical situation briefing for GV-YOLO26, formatted for in-depth analysis, research audit, and evaluation on web reasoning platforms (Argon / NotebookLM / LLM workspaces).  
> **Repository Path**: `/home/benny/Projects/GV-YOLO26`  
> **Timestamp / Project State**: October 2026 | Active / Submission-Ready & Extended Evolution Stage

---

## 1. Executive Summary & Project Identity

**GV-YOLO26** is an ultra-lightweight, end-to-end, Non-Maximum Suppression (NMS)-free neural network engineered to detect roadway surface hazards—specifically **speed bumps** and **potholes**—from forward-facing monocular cameras for active vehicle suspension preview control.

* **Academic Degree / Course**: U23ADS705 – Mini Project, B.Tech Artificial Intelligence and Data Science
* **Institution**: Department of Information Technology, Sona College of Technology (Autonomous), Anna University, Salem, Tamil Nadu, India
* **Student Researchers**:
  * **Benadict Infant A** (Reg. No. `61782323110022`)
  * **Elangovan S** (Reg. No. `61782323110035`)
* **Project Supervisor**: **Dr. P. Shanmugaraja** (Assistant Professor, Dept. of Information Technology)
* **Head of Department**: **Dr. J. Akilandeswari** (Professor & Head, Dept. of Information Technology)
* **Conference Paper Derivative**: *"Comparative Evaluation of YOLOv8 to YOLO26 and RT-DETR for Speed Bump and Pothole Detection"* (Drafted and revised conference manuscript)
* **Core Artifact Deliverables**:
  * Formal College Project Report: [GV_YOLO26_Project_Report.docx](file:///home/benny/Projects/GV-YOLO26/GV_YOLO26_Project_Report.docx) (78 pages, 4.0 MB) & [GV_YOLO26_Project_Report.pdf](file:///home/benny/Projects/GV-YOLO26/GV_YOLO26_Project_Report.pdf) (2.9 MB).
  * Real-time Interactive Web HUD Demo: [run_demo.py](file:///home/benny/Projects/GV-YOLO26/run_demo.py) & [demo_app/](file:///home/benny/Projects/GV-YOLO26/demo_app/).
  * Reproducible Kaggle Training Notebooks: [GV-YOLO26/Kaggle_Train_GV_YOLO26.ipynb](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/Kaggle_Train_GV_YOLO26.ipynb), [generate_simple_nb.py](file:///home/benny/Projects/GV-YOLO26/generate_simple_nb.py), [build_kaggle_nb.py](file:///home/benny/Projects/GV-YOLO26/build_kaggle_nb.py).

---

## 2. Engineering Motivation & Vehicle Control Problem

### 2.1 The Preview Active Suspension Imperative
Traditional automotive suspensions react passively after wheel impact, transmitting sharp accelerations into the cabin and chassis. Active and semi-active suspension systems (e.g., magnetorheological or electronically controlled continuously variable dampers) can preemptively adjust damping stiffness or ride height if road anomalies are detected **15 to 30 meters ahead** (0.5 to 1.0 seconds of preview window at 50–100 km/h).

### 2.2 Why Existing Detectors Fail in Embedded Automotive Preview
1. **The Post-Processing Bottleneck (NMS Latency Jitter)**: Standard YOLO variants (YOLOv8, YOLOv10, YOLO11) rely on non-maximum suppression (NMS) to eliminate redundant anchor boxes. NMS runtime is non-deterministic ($\mathcal{O}(N^2)$ in dense scenarios), causing frame drops and jitter in hard real-time automotive Electronic Control Units (ECUs).
2. **Small / Distant Object Degradation**: Potholes appear minute (46.5% of instances are $<32\times32$ pixels at $640\times640$ resolution). Aggressive downsampling in standard stride-8/16/32 networks loses spatial features before reaching detection heads.
3. **Severe Resource Constraints**: In-cabin mobile devices or cost-sensitive automotive micro-controllers cannot support heavy feature pyramids or 30M+ parameter vision transformers (e.g., RT-DETR).

---

## 3. Architectural Design of GV-YOLO26

GV-YOLO26 uses **YOLO26n** as its foundational backbone and systematically re-engineers it for low compute and small-hazard sensitivity.

```
                      [ Input Image: 640 x 640 x 3 ]
                                    │
                       Stem / Low-level Backbone
                                    ├──────────────────────────┐ (High Resolution)
                                    │                          │
                               Stage 2 (P2: 160x160)           │
                                    │                          │
                   GhostConv Strided Downsampling (Cheap)      │
                                    │                          │
                         Stage 3 (P3: 80x80)                   │
                                    │                          │
                   GhostConv Strided Downsampling              │
                                    │                          │
                         Stage 4 (P4: 40x40)                   │
                                    │                          │
                  VoVGSCSPC Feature Mixing (Cheap)             │
                                    │                          │
                         Stage 5 (P5: 20x20)                   │
                                    │                          │
                        SPPF / Context Fusion                  │
                                    │                          │
                   ┌────────────────┴────────────────┐         │
                   │    VoVGSCSPC Neck Aggregation   │         │
                   └────────────────┬────────────────┘         │
                                    │                          │
              ┌───────────────┬─────┴─────────┬────────────────┘
              ▼               ▼               ▼                ▼
         P5 (20x20)      P4 (40x40)      P3 (80x80)       P2 (160x160)
         Large Bump      Medium Hazard   Standard Box     Distant Pothole
              └───────────────┼───────────────┴────────────────┘
                              ▼
        [ LGDH: Lightweight Grouped Detection Head ]
        - Grouped Convolutions (channels_per_group=8)
        - Dual-Branch One-to-One Matching (NMS-Free End-to-End)
        - CIoU + Normalized Wasserstein Distance (NWD) Loss
                              │
                              ▼
            [ Final Bounding Boxes + Confidence ]
                   (Pothole, Speed Bump)
```

### Key Technical Mechanisms

| Module | Location | Source Implementation | Mechanism & Rationale |
| :--- | :--- | :--- | :--- |
| **YOLO26n Base** | Core Engine | Ultralytics `8.4.156` | Provides end-to-end NMS-free dual-branch one-to-one matching; removes DFL post-processing overhead. |
| **P2 Stride-4 Branch** | Backbone & Head | [gv-yolo26.yaml](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/models/gv-yolo26.yaml) | Adds a $160 \times 160$ feature map level to preserve fine spatial boundaries of distant potholes before downsampling. |
| **LGDH Head** | Detection Head | [head.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/head.py) | **Lightweight Grouped Detection Head**: Uses grouped convolutions (`channels_per_group=8`) in head stems to slash the high FLOP cost imposed by the $160\times160$ P2 grid. |
| **GhostConv** | Backbone/Neck | [blocks.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/blocks.py) | Generates intrinsic feature maps with cheap linear operations ($1\times1$ primary conv + $5\times5$ depthwise conv), replacing standard downsampling convs. |
| **VoVGSCSPC** | Deep Backbone/Neck | [blocks.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/blocks.py) | Replaces heavy C3k2 blocks. Integrates GSConv with channel shuffle and cross-stage partial connections to maximize spatial feature reuse at low FLOPs. |
| **NWD Box Loss** | Training Loss | [loss.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/loss.py) | $\mathcal{L}_{\text{NWD}} = 1 - \exp\left(-\frac{\sqrt{W_2^2(A, B)}}{C}\right)$. Mitigates high gradient sensitivity of strict IoU on tiny boxes with zero intersection. |
| **Weather Augmentation** | Input Pipeline | [augment.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/augment.py) | Albumentations pipeline simulating real-world driving: synthetic rain, snow, motion blur, and severe illumination changes. |
| **Temporal Tracker** | Video Pipeline | [demo_video.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/demo_video.py) | ByteTrack tracker + 3-frame confirmation rule ($\text{hits} \ge 3, \text{conf} \ge 0.35$) to eliminate flickering false alarms from road paint/shadows. |

---

## 4. Empirical Benchmark & Ablation Findings

### 4.1 Stock Detector Benchmark (Base Model Selection)
Before architectural customization, 7 stock object detectors were trained from scratch on the SBP-YOLO road dataset under identical conditions (Table 5.1 of report):

| Model | Parameters | GFLOPs (640) | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Latency (T4 FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8n** | 3.01 M | 8.1 | 71.4% | 33.6% | 76.5% | 68.2% | 39.8 FPS (25.1 ms) |
| **YOLOv9t** | 1.99 M | 7.6 | 73.0% | 34.7% | 78.4% | 69.8% | 36.1 FPS (27.7 ms) |
| **YOLOv10n** | 2.27 M | 6.5 | 72.8% | 34.2% | 77.9% | 69.1% | 38.0 FPS (26.3 ms) |
| **YOLO11n** | 2.59 M | 6.4 | 73.9% | 35.1% | 79.1% | 70.4% | 39.5 FPS (25.3 ms) |
| **YOLO12n** | 2.59 M | 6.5 | 73.6% | 35.0% | 78.8% | 70.1% | 32.4 FPS (30.9 ms) |
| **YOLO26n (Selected)** | **2.38 M** | **5.3** | **74.8%** | **35.5%** | **82.3%** | **71.2%** | **58.0 FPS (17.2 ms)** |
| **RT-DETR-L** | 31.99 M | 103.4 | 67.5% | 31.8% | 71.0% | 64.2% | 12.2 FPS (82.0 ms) |

**Conclusion**: YOLO26n established superior accuracy, lowest computation (5.3 GFLOPs), highest precision (82.3%), and fastest native inference (58 FPS).

> [!NOTE]
> **Baseline Framing & Discrepancy Reconciliation**:
> * **74.8% mAP@0.5 (Stock Detector Benchmark, Table 5.1)**: Represents the exploratory initial benchmark where off-the-shelf architectures (YOLOv8n through YOLO26n and RT-DETR-L) were trained from scratch for 100 epochs on the uncurated raw Kaggle/Roboflow dataset split prior to data-leakage auditing.
> * **81.0% mAP@0.5 (Stage 1 Ablation Baseline, Table 5.3)**: Represents the rigorously standardized ablation baseline trained for 150 epochs on the audited `train_used` partition with early-stopping selection on `val_carved`, and evaluated strictly on the leakage-free `clean_holdout` set.

---

### 4.2 The 6-Stage Ablation Ladder
Each innovation was introduced incrementally, trained for 150 epochs on Kaggle Tesla T4 GPUs, and audited on a leakage-free held-out dataset:

| Stage | Configuration | Parameters | GFLOPs | Checkpoint | Clean mAP@0.5 | Clean mAP@0.5:0.95 | T4 FP32 FPS |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **s1** | **YOLO26n Baseline** | 2.38 M | 5.3 | 5.1 MB | 81.0% | 48.6% | 58.0 FPS |
| **s2** | **+ P2 Head (Stride-4)** | 2.40 M | 6.6 | 5.3 MB | 80.3% | 47.6% | 55.7 FPS |
| **s3** | **+ LGDH Head** | 2.37 M | 6.2 | 5.2 MB | 80.0% | 46.5% | 57.4 FPS |
| **s4** | **+ GhostConv** | 2.08 M | 5.8 | 4.6 MB | 81.5% | 47.2% | 56.1 FPS |
| **s5** | **+ VoVGSCSPC** | 1.57 M | 5.1 | 3.7 MB | 77.9% | 45.1% | 59.2 FPS |
| **s6** | **+ NWD Loss (GV-YOLO26)** | **1.57 M** | **5.1** | **3.7 MB** | **77.5%** | **45.8%** | **57.5 FPS** |

#### Critical Ablation Insights:
1. **Compute Squeeze Succeeded**: GV-YOLO26 achieved a **33.7% parameter reduction** (1.57M vs 2.38M) and cut FLOPs to 5.1 GFLOPs, despite managing an extra high-resolution $160\times160$ detection grid.
2. **Batch=1 Latency Reality**: Single-frame latency on GPUs remained stable (~57.5 FPS) because GPU kernel launch overhead and intermediate tensor memory traffic for the $160\times160$ P2 grid balanced out arithmetic FLOP savings.
3. **The Accuracy vs Capacity Trade-off**: On this specific dataset, speed bumps are wide targets (median width: 72% of image), relying on deep semantic receptive fields. Potholes gained from P2 resolution, but replacing standard convolutions with cheap VoVGSCSPC blocks reduced deep layer representational capacity, creating a slight overall accuracy decrease on the strict held-out set (-3.5% mAP50 vs base).

---

### 4.3 Pareto Operating Points & Deployed Production Variants
Rather than viewing Stage 6 and subsequent high-accuracy models as conflicting, the project formalizes three distinct Pareto operating tiers tailored to specific automotive deployment environments:

1. **Tier A: Ultra-Compact Edge Tier (`GV-YOLO26 Stage 6`, `models/gv-yolo26n.yaml`)**:
   * **1.57M Parameters | 5.1 GFLOPs | 3.7 MB checkpoint | 77.5% mAP@0.5** (Clean Holdout).
   * **Target**: Highly constrained edge microcontrollers, auxiliary smart dashcam sensors, and low-power embedded processors where memory footprint and minimum parameter size are paramount.
2. **Tier B: Balanced Automotive Production Tier (`GV-YOLO26-Opt` / `GV-YOLO26-Simple`, `weights/gv_yolo26_simple_best.pt`)**:
   * **2.08M Parameters | 5.3–5.8 GFLOPs | 4.5 MB checkpoint | 82.8%–84.1% mAP@0.5** (93.3% Speed Bump, 72.3% Pothole).
   * **Target**: Production automotive active suspension preview Electronic Control Units (ECUs). Preserves full C3k2 representational capacity at P5 for wide road speed bumps while integrating GhostConv downsampling and VoVGSCSPC spatial mixing in intermediate stages.
3. **Tier C: Hybrid Feature Injection Variant (`GV-YOLO26-P2P3`, `models/gv-yolo26-p2p3.yaml`)**:
   * **2.01M Parameters | 4.9 GFLOPs | 4.6 Fused GFLOPs | 8,400 Detection Anchors**.
   * **Target**: Avoids the $25,600$ anchor dilution of a 4th P2 detection head by reorganizing P2 ($160\times160$) features directly into P3 ($80\times80$) via Space-to-Depth (`SPDConv`) and channel concatenation, retaining standard 3-scale one-to-one matching.
4. **Tier D: Enhanced Edge-Case Detector (`GV-YOLO26-X`, `models/gv-yolo26-x.yaml`)**:
   * **2.11M Parameters (1.98M Fused) | 5.4 GFLOPs (4.8 Fused) | 8,400 Detection Anchors | NMS-Free Detect Head**.
   * **Target**: Designed to match SBP-YOLO backbone detection quality while guaranteeing deterministic NMS-free real-time latency. Combines P2-to-P3 `SPDConv` spatial injection for tiny potholes, preserves full `C3k2` receptive field at P5 for wide speed bumps, and introduces a dedicated dusk/low-contrast augmentation pipeline (`dusk_augmentations()` in `gv_modules/augment.py`) to resolve dark asphalt and faded marking edge cases without requiring additional training data.
* **External Reference Benchmark (SBP-YOLO by Liang et al.)**:
  * 2.62M Parameters, 6.6 GFLOPs, 86.9% mAP@0.5 (based on YOLOv11n with teacher distillation and 4-scale NMS head).

---

## 5. Dataset Engineering & Visual Data Leakage Resolution

### 5.1 The Data Leakage Audit
The SBP-YOLO dataset ([speed12](https://www.kaggle.com/datasets/elangovan001/speed12) on Kaggle) consists of 7,600 images (12,205 potholes, 3,217 speed bumps). An automated audit conducted in [tools/make_clean_split.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/tools/make_clean_split.py) revealed severe train-test contamination:
1. **Augmented Duplicates**: 526 of the 1,125 raw test images and 378 valid images were noise-, blur-, or crop-augmented clones of training images (sharing identical Roboflow hashes like `add_gasuss_noise_1386_jpg.rf.df5cd...` vs `1386_jpg.rf.df5cd...`).
2. **Drive Sequence Overlap**: 839 test images and 554 valid images were consecutive GoPro dashcam frames (`G00xxxxx.jpg`, `DASH_CAM...`) sharing visual scenes with the training split.

### 5.2 Resolution & Clean Held-Out Partition
All contaminated images were programmatically purged. The resulting partition:
* **`train_used`**: 5,243 clean training images (8,300 potholes, 2,144 speed bumps).
* **`val_carved`**: 586 images (10% seeded sample of training split, seed 42) for checkpoint selection.
* **`clean_holdout`**: **482 strictly independent evaluation images** (286 from test, 196 from valid; 354 potholes, 384 speed bumps) sharing zero source photos, videos, or drives with training.
* **Leakage Impact**: Raw test split evaluation inflated mAP by **2 to 4 percentage points** (e.g., GV-YOLO26 scored 82.1% on raw test vs 77.5% on clean held-out), proving the scientific necessity of the clean split.

---

## 6. The Knowledge Distillation Negative Result & GIoU Divergence

Knowledge distillation was attempted using an S-scale teacher model (**GV-YOLO26-S**: 6.11M params, 19.2 GFLOPs, 77.7% mAP50):
* **Formulation**: Combined Bridging Cross-task Knowledge Distillation (BCKD) with Channel-Wise Distillation (CWD).
* **Observed Failure Modes**:
  1. *Scratch Training (150 epochs)*: Diverged at epoch 3 during learning-rate warmup; classification loss spiked to $>10$, and loss became `NaN`.
  2. *Fine-tuning Stage 6 (50 epochs)*: Degraded validation mAP from 81.3% down to 25.5% within 1 epoch; distillation loss became negative (reaching $-1260$).
* **Root-Cause Discovery**:
  * YOLO26 is an end-to-end model that decodes box distances without Distribution Focal Loss (DFL) and lacks strict coordinate positivity constraints on raw head outputs.
  * During initial convergence, approximately **0.9% of student raw predictions decoded to negative widths or heights**.
  * When fed into Generalized IoU (GIoU), this caused GIoU to range from $-10$ to $+1$ rather than remaining in $[-1, 1]$.
  * Consequently, the loss term $(1 - \text{GIoU})$ was unbounded below, causing gradient descent to drive the loss to negative infinity and destroy detector features.

### 6.1 Mathematical Resolution & Stabilization (October 2026 Upgrade)
To eliminate divergence while preserving end-to-end NMS-free behavior, three complementary safeguards were formulated and implemented:
1. **Coordinate Positivity Clamping**: Raw predicted distance logits $(lt, rb)$ are projected via $F.\text{softplus}(\text{pred\_dist}, \beta=1.0).\text{clamp}(\min=10^{-4})$ prior to anchor box decoding (`dist2bbox`), mathematically guaranteeing positive widths $w = x_2 - x_1 > 0$ and heights $h = y_2 - y_1 > 0$ across 100% of anchors.
2. **Hybrid Bounded Distillation Loss**: Replaced unbounded GIoU with a convex combination of clamped CIoU/GIoU and Normalized Wasserstein Distance (NWD):
   $$\mathcal{L}_{\text{box\_distill}} = (1.0 - \alpha) \cdot (1.0 - \text{CIoU}_{\text{clamped}}) + \alpha \cdot \mathcal{L}_{\text{NWD}}$$
   Because $\mathcal{L}_{\text{NWD}} = 1 - \exp\left(-\frac{\sqrt{W_2^2}}{C}\right) \in [0, 1)$ is strictly non-negative and bounded, it prevents gradient explosion under uncalibrated student predictions.
3. **Teacher Foreground Masking & Warmup**: Distillation gradients on uncalibrated background queries are suppressed by masking against teacher confidence ($\text{conf}_{\text{teacher}} > 0.25$), accompanied by a 3-epoch linear distillation weight warmup.

---

## 7. Active Suspension Preview Physics, Adaptive Tracker & HUD Web App

### 7.1 Preview Physics & Monocular Geometry
Active and semi-active vehicle suspensions require previewing road anomalies 15 to 30 meters ahead (0.5 to 1.0 s preview window at 50–100 km/h) to adjust damping valves prior to tire contact. Monocular distance and Time-to-Impact ($\text{TTI}$) are estimated using pinhole ground-plane projection:
* **Longitudinal Distance ($Z$)**:
  $$Z = \frac{h_{\text{cam}}}{\tan(\theta_{\text{pitch}} + \alpha_v)}, \quad \alpha_v = \arctan\left(\frac{v_{\text{bottom}} - c_y}{f_y}\right)$$
  where $h_{\text{cam}} = 1.25\,\text{m}$, $\theta_{\text{pitch}} = 8.0^\circ$, and $\text{VFOV} = 50.0^\circ$, clamped to the $[2.0\,\text{m}, 45.0\,\text{m}]$ suspension actuation envelope.
* **Time-to-Impact ($\text{TTI}$)**:
  $$\text{TTI} = \frac{Z}{v_{\text{ego}} / 3.6} \times 1000 \quad [\text{ms}]$$
* **Proactive Suspension Damping Commands**:
  * `"SOFTEN_FAST"`: Triggered when $15.0 \le Z \le 30.0\,\text{m}$ to rapidly open damper bypass valves and absorb impact.
  * `"PRELOAD_CRITICAL"`: Triggered when $Z < 15.0\,\text{m}$ in near-field critical zone to brace against suspension bottoming-out.
  * `"NOMINAL"`: Default cruising stiffness when road ahead is clear.

### 7.2 FPS- & Confidence-Adaptive Hysteresis Tracker
Replaces the static 3-frame delay with an adaptive dual-criterion confirmation rule:
* **Immediate 1-Frame Fast Trigger**: If single-frame detection confidence is high ($\text{conf} \ge 0.72$) and the box is situated in the lower 40% of the frame (near-field threat).
* **Cumulative Confidence Hysteresis**: Otherwise, confirms when cumulative confidence across tracked frames reaches $\sum \text{conf} \ge 1.05$ (equivalent to 3 frames at 0.35 confidence), preserving robustness against distant flickering anomalies.

### 7.3 Mobile HUD Interactive Demo Application
Located in [demo_app/](file:///home/benny/Projects/GV-YOLO26/demo_app/) and executed via [run_demo.py](file:///home/benny/Projects/GV-YOLO26/run_demo.py):
* **Backend**: FastAPI with async image streaming, ByteTrack tracking, monocular preview telemetry, and OpenCV HUD overlays.
* **Frontend**: Responsive single-page HTML5/CSS3 application with Web Audio API chime synthesis, live 4-card telemetry (FPS, Active Hazard, Metric Preview $Z$/TTI, Damping Command), and 4-speed ego velocity selector (40 / 60 / 80 / 100 km/h).
* **Multi-Model Comparison**: Real-time 1-tap switching between `GV-YOLO26`, `GV-YOLO26-Opt`, and `SBP-YOLO`.

---

## 8. Current Situation & Technical Constraints

### 8.1 Strictly Pinned Dependencies
* **`ultralytics==8.4.156`**: Pinned explicitly. In [gv_modules/register.py](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/register.py), the framework monkey-patches `ultralytics.nn.tasks.parse_model` at runtime using exact source code anchors (`base_end`, `repeat_end`, `detect_end`). Any upgrade to Ultralytics without adapting these patch anchors will break layer parsing.
* **Python Virtual Environment**: Located at `/home/benny/Projects/GV-YOLO26/venv` (Python 3.11.16, PyTorch 2.14.1, CUDA 13 bindings). Note that invoking `./venv/bin/pip` directly fails due to relocated directory shebangs; always invoke using `python -m pip`.

### 8.2 Current File & Repository Structure

```
/home/benny/Projects/GV-YOLO26/
├── AGENTS.md                                   # Persistent operational instructions & commands
├── context.md                                  # This master context and briefing document
├── summary.md                                  # Detailed project execution log & tables
├── run_demo.py                                 # Launcher for mobile HUD demo (port 8000)
├── build_kaggle_nb.py                          # Generator for GV-YOLO26-Opt Kaggle notebook
├── generate_simple_nb.py                       # Generator for GV-YOLO26-Simple Kaggle notebook
├── GV_YOLO26_Project_Report.docx               # 78-page College Project Report (Word)
├── GV_YOLO26_Project_Report.pdf                # Compiled Project Report (PDF)
├── Speed_Bump_YOLO26_Conference_Paper_Revised(1).docx  # Revised IEEE-format conference manuscript
│
├── demo_app/                                   # Real-time HUD web application
│   ├── app.py                                  # FastAPI backend with ByteTrack
│   ├── sbp_modules.py                          # Runtime registration for SBP-YOLO modules
│   └── static/index.html                       # Mobile HUD frontend
│
├── GV-YOLO26/                                  # Core machine learning package
│   ├── gv_modules/                             # Custom architecture & training modules
│   │   ├── register.py                         # Monkey-patch runtime layer registration
│   │   ├── blocks.py                           # GhostConv, GSConv, VoVGSCSPC
│   │   ├── head.py                             # LGDH (Lightweight Grouped Detection Head)
│   │   ├── loss.py                             # NWDDetectionLoss & nwd calculation
│   │   ├── augment.py                          # Albumentations adverse weather pipeline
│   │   ├── trainer.py                          # GVTrainer, NWDTrainer, DistillTrainer
│   │   └── model.py                            # GVDetectionModel
│   ├── models/                                 # Network architecture YAML files
│   │   ├── gv-yolo26.yaml                      # Final 6-stage architecture (N and S scales)
│   │   ├── gv-yolo26-simple.yaml               # Pure backbone-modified variant (82.8% mAP)
│   │   ├── gv-yolo26-opt.yaml                  # Optimized augmentations variant (84.1% mAP)
│   │   └── ablation/                           # s2_p2, s3_lgdh, s4_ghost YAMLs
│   ├── tools/
│   │   ├── benchmark.py                        # Automated accuracy, latency, and FLOP profiler
│   │   ├── make_clean_split.py                 # Visual leakage elimination tool
│   │   └── model_summary.py                    # Multi-stage model verification tool
│   ├── train.py                                # Master training entrypoint
│   ├── distill.py                              # Knowledge distillation script
│   ├── val.py                                  # Test split evaluation and latency script
│   ├── infer.py                                # Single image/video inference
│   ├── demo_video.py                           # Video tracking demo with ByteTrack
│   ├── data_local/                             # Leakage-free split definitions & logs
│   │   ├── data_clean.yaml                     # Points to 482-image clean held-out set
│   │   └── clean_holdout.txt                   # Verified image file list
│   └── weights/
│       └── gv_yolo26_simple_best.pt            # Active high-performance checkpoint
│
└── report_build/                               # Automated report generator
    ├── build_report.py                         # Docx assembler from experiment JSONs
    ├── docxlib.py                              # Low-level OpenXML DOM manipulator
    ├── narrative.py                            # Report text and narrative binding
    └── inputs/                                 # Raw benchmark, dataset, and video JSONs
```

---

## 9. Strategic Questions & Next Analysis Horizons for Argon

When analyzing this project in web reasoning environments, key technical areas of focus include:

1. **Theoretical Convergence of Distillation on Anchor-Free / NMS-Free YOLO**:
   * *Problem*: How to formally reformulate BCKD and CWD distillation losses to guarantee coordinate positivity and prevent gradient explosion in the absence of DFL distributions.
2. **Quantization Degradation Under Post-Training Quantization (PTQ)**:
   * *Problem*: Preliminary INT8 conversion dropped mAP from 45.8% to 35.0%. How to structure Quantization-Aware Training (QAT) in PyTorch to prevent precision loss in depthwise convolutions of GhostConv and GSConv.
3. **Optimizing Stride-4 P2 Latency vs Feature Gain**:
   * *Problem*: The P2 grid provides fine-grained pothole localization but consumes significant memory bandwidth at batch=1. Can depthwise separable grouped convolutions or attention gating further compress P2 without sacrificing small-box recall?
4. **Physical Interfacing with Electronic Suspension Systems**:
   * *Problem*: What are the latency bounds and predictive time-to-impact (TTI) formulations required to bridge 57 FPS monocular vision output with vehicular CAN bus damper actuation?
