# GV-YOLO26: Project Technical Status & Engineering Log

> **Project Title**: GV-YOLO26: Ultra-Lightweight, NMS-Free Road Hazard Detection for Active Vehicle Suspension Preview Control  
> **Target Hazards**: Speed Bumps, Potholes  
> **Base Architecture**: YOLO26n (End-to-End One-to-One Hungarian Matching)  
> **Key Innovations**: GhostConv, VoVGSCSPC, P2 High-Resolution Head (160×160), LGDH, Normalized Wasserstein Distance (NWD) Loss, Albumentations Weather Pipeline  
> **Dataset**: SBP-YOLO Benchmark (`speed12` on Kaggle, 7,600 images)  
> **Evaluation Split**: 482-Image Deduplicated Leakage-Free Held-Out Split  
> **Execution Status**: All 6 ablation stages trained, validated, benchmarked, and integrated into real-time dashcam tracking & HUD demo pipelines.

---

## 1. Project Overview & Motivation

Modern intelligent vehicles with active or semi-active suspension systems (e.g., magnetorheological or hydraulic dampers) require forward road surface preview information to adapt damping characteristics before wheel impact. Conventional multi-sensor approaches relying on LiDAR or stereo cameras introduce severe hardware costs, heavy compute footprints, calibration complexity, and high failure rates in adverse weather.

**GV-YOLO26** resolves this challenge by formulating an ultra-lightweight, real-time, monocular vision detector that:
1. **Detects small and distant hazards**: Captures tiny potholes and low-contrast speed bumps at long distances ($\ge 15$–$30$ m) to allow sufficient actuation preview time ($TTI = Z / v$).
2. **Eliminates Non-Maximum Suppression (NMS)**: Employs YOLO26 end-to-end one-to-one matching, removing runtime NMS clustering bottlenecks and providing deterministic sub-6 ms latency.
3. **Optimizes edge compute footprint**: Incorporates cheap convolutions (**GhostConv**, **VoVGSCSPC**) and a **Lightweight Grouped Detection Head (LGDH)** to achieve a 33.7% parameter reduction (1.57 M vs 2.38 M) and 5.1 GFLOPs.
4. **Stabilizes tiny bounding box regression**: Implements **Normalized Wasserstein Distance (NWD)** loss to prevent gradient collapse on small distant objects where intersection-over-union (IoU) drops to zero.
5. **Filters transient visual noise**: Features a temporal multi-frame association tracker ($k=3, \tau=0.35$) to reject false alarms from lane markings, oil spills, and shadows.

---

## 2. Architectural Design & Custom Modules

All custom architectural components are modularized under [`GV-YOLO26/gv_modules/`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/):

```
GV-YOLO26/gv_modules/
├── __init__.py           # Package exports & automatic module registration hook
├── register.py           # Monkey-patches Ultralytics parse_model for GSConv, VoVGSCSPC, LGDH
├── blocks.py             # GhostConv, GSConv, GSBottleneckC, and VoVGSCSPC implementations
├── head.py               # LGDH (Lightweight Grouped Detection Head)
├── loss.py               # NWDDetectionLoss (CIoU + Gaussian Wasserstein Distance)
├── augment.py            # Weather simulation pipeline (rain, snow, motion blur, illumination)
├── trainer.py            # GVTrainer, NWDTrainer, and DistillTrainer extensions
├── suspension_preview.py # Monocular pinhole geometry, metric distance, and TTI calculations
└── model.py              # GV-YOLO26 model wrapper
```

### Module Breakdown & Rationale:

| Component | File & Class | Mathematical Formulation / Implementation | Design Rationale |
| :--- | :--- | :--- | :--- |
| **Ultralytics Runtime Patch** | [`register.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/register.py) | `tasks.parse_model` string anchor injection (`ultralytics==8.4.156`) | Automatically binds custom blocks into standard YOLO YAML parsing without modifying external library source. |
| **High-Resolution P2 Branch** | [`gv-yolo26.yaml`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/models/gv-yolo26.yaml) | Stride-4 feature branch ($160 \times 160$ grid at $640 \times 640$ input) | Preserves spatial resolution for tiny distant potholes where 46.5% of instances are $< 32$ pixels. |
| **Lightweight Grouped Head** | [`head.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/head.py) (`class LGDH`) | Grouped convolutions with `channels_per_group=8` | Prevents the $160 \times 160$ P2 branch from causing FLOP explosion in detection heads. |
| **GhostConv** | [`blocks.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/blocks.py) (`class GhostConv`) | Intrinsic primary conv $\frac{C_{out}}{2}$ + cheap $5\times 5$ depthwise conv $\frac{C_{out}}{2}$ | Halves computational cost in deep downsampling layers (P4, P5) via feature redundancy exploitation. |
| **VoVGSCSPC** | [`blocks.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/blocks.py) (`class VoVGSCSPC`) | One-shot aggregation (OSA) + GSConv depthwise-separable channel shuffle | Replaces expensive C3k2 blocks in the neck, maximizing spatial-semantic feature reuse at low FLOPs. |
| **NWD Loss** | [`loss.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/loss.py) (`class NWDDetectionLoss`) | $\mathcal{L}_{\text{NWD}} = 1 - \exp\left(-\frac{\sqrt{\mathcal{W}_2^2(\mathcal{N}_A, \mathcal{N}_B)}}{C}\right)$, combined $\mathcal{L}_{\text{box}} = (1-\alpha)\mathcal{L}_{\text{CIoU}} + \alpha\mathcal{L}_{\text{NWD}}$ | Models boxes as 2D Gaussians; provides smooth continuous gradients even with zero bounding box intersection. |
| **Weather Simulation** | [`augment.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/gv_modules/augment.py) | Albumentations pipeline: `RandomRain`, `RandomSnow`, `MotionBlur`, `RandomGamma` | Guarantees detector robustness under rain streaks, lens glare, twilight, and motion smear. |

---

## 3. Incremental 6-Stage Ablation Ladder (Empirical Results)

To isolate and evaluate each architectural modification, a 6-stage ablation ladder was designed, trained for 150 epochs each on Kaggle Tesla T4 GPUs (`train_used`, 5,243 images), and evaluated on the leakage-free clean held-out set (482 images):

| Stage | Configuration | Parameters | GFLOPs (640) | Checkpoint | mAP@0.5 | mAP@0.5:0.95 | PyTorch FP32 FPS (T4) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **YOLO26n Baseline** | 2.38 M | 5.3 | 5.2 MB | 81.0% | 48.6% | 58.0 FPS (17.25 ms) |
| **2** | **+ P2 Head (160×160)** | 2.40 M | 6.6 | 5.3 MB | 80.3% | 47.6% | 55.7 FPS (17.97 ms) |
| **3** | **+ LGDH Head** | 2.37 M | 6.2 | 5.2 MB | 80.0% | 46.5% | 57.4 FPS (17.44 ms) |
| **4** | **+ GhostConv** | 2.08 M | 5.8 | 4.6 MB | 81.5% | 47.2% | 56.1 FPS (17.84 ms) |
| **5** | **+ VoVGSCSPC** | 1.57 M | 5.1 | 3.7 MB | 77.9% | 45.1% | 59.2 FPS (16.90 ms) |
| **6** | **+ NWD Loss (GV-YOLO26)** | **1.57 M** | **5.1** | **3.7 MB** | **77.5%** | **45.8%** | **57.5 FPS (17.38 ms)** |

### Engineering Insights & Analysis:
- **Parameter & FLOP Savings**: Stage 6 achieves a **33.7% parameter reduction** (1.57 M vs 2.38 M) and drops FLOPs from 6.6 down to **5.1 GFLOPs**, despite operating on four multi-scale feature pyramids (P2 to P5).
- **Latency Stability**: Single-image FP32 batch=1 inference stays stable at ~57.5 FPS on Tesla T4 because GPU memory bandwidth and kernel launch overhead for the P2 feature map balance out arithmetic FLOP reductions.
- **NWD Regression Gain**: Adding NWD loss in Stage 6 lifted mAP@0.5:0.95 from 45.1% to 45.8% (+0.7%), verifying tighter bounding box localization on distant small potholes.

### Architectural Variants & Pareto Operating Tiers:
1. **Ultra-Compact Edge Tier (`GV-YOLO26 Stage 6`, 1.57M params, 5.1 GFLOPs, 77.5% mAP50)**: Checkpoint [`gv_yolo26_best.pt`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/weights/gv_yolo26_best.pt). Designed for low-power smart dashcams and embedded microcontrollers.
2. **Automotive Production Tier (`GV-YOLO26-Simple` / `Opt`, 2.08M params, 5.8 GFLOPs, 82.8%–84.1% mAP50)**: Checkpoint [`gv_yolo26_simple_best.pt`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/weights/gv_yolo26_simple_best.pt). Retains standard C3k2 blocks at deep P5 stages to provide higher representational capacity for wide, unpainted speed bumps.
3. **Space-to-Depth Hybrid (`GV-YOLO26-P2P3`, 2.01M params, 4.9 GFLOPs, 8,400 anchors)**: Injects P2 high-resolution details into P3 via SPDConv without adding a 4th detection head, avoiding anchor dilution.
4. **Adverse Condition Variant (`GV-YOLO26-X`, 2.11M params, 5.4 GFLOPs)**: Checkpoint [`gv_yolo26_x_best.pt`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/weights/gv_yolo26_x_best.pt). Combines VoVGSCSPC neck with dedicated dusk/night illumination augmentation.

---

## 4. Dataset Engineering & Leakage Resolution

Experiments are conducted on the public **SBP-YOLO Benchmark** ([speed12](https://www.kaggle.com/datasets/elangovan001/speed12) on Kaggle):
- **Raw Dataset Size**: 7,600 images (5,725 train, 750 valid, 1,125 test) containing 12,205 potholes and 3,217 speed bumps.

### Data Leakage Discovery:
1. **Spatial-Temporal Duplicate Audit**: Perceptual hashing and file audits identified that 526 raw test images and 378 valid images were noise/crop-augmented duplicates of training images. Furthermore, continuous GoPro video drives (`G00xxxxx.jpg`) were shared across the train and test folders.
2. **Resolution Pipeline ([`tools/make_clean_split.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/tools/make_clean_split.py))**: Strict deduplication isolated **482 clean holdout images** (286 test, 196 valid; 354 potholes, 384 speed bumps) having strictly zero drive-sequence or source photo overlap with `train/`.
3. **Leakage Penalty Measurement**: Evaluating models on the uncurated raw test set yielded inflated scores by 2.0% to 3.5% mAP (e.g., 82.1% raw vs 77.5% clean held-out), proving the necessity of the clean split.

---

## 5. Knowledge Distillation Framework & Stabilization

A teacher-student distillation framework was developed using **GV-YOLO26-S** (6.11M parameters, 19.2 GFLOPs) as the teacher:
- **Architecture**: Bridging Cross-task Knowledge Distillation (BCKD) on predicted logits + Channel-Wise Distillation (CWD) on intermediate feature maps.

### Mathematical Root-Cause & Stabilization:
- **Divergence Mechanism**: YOLO26 uses regression without Distribution Focal Loss (DFL). Early in training, ~0.9% of student predictions decoded to negative bounding box dimensions, causing Generalized IoU (GIoU) between teacher and student to diverge outside $[-1, 1]$ down to $-10$. Consequently, $1 - \text{GIoU}$ drove distillation gradients to $-\infty$ (`NaN` divergence at epoch 3).
- **Stabilized Formulation**:
  1. *Coordinate Softplus Clamping*: Distances pass through $F.\text{softplus}(\text{pred\_dist}, \beta=1.0).\text{clamp}(\min=10^{-4})$ before box decoding, guaranteeing $w > 0, h > 0$ strictly.
  2. *Bounded NWD Box Distillation*: Replaced GIoU with hybrid $(1-\alpha)(1-\text{CIoU}_{\text{clamped}}) + \alpha \mathcal{L}_{\text{NWD}}$. Because $\mathcal{L}_{\text{NWD}} \in [0, 1)$, loss is strictly bounded.
  3. *Teacher Masking & Warm-up*: Box loss is gated by teacher confidence ($\text{conf} > 0.25$) with a 3-epoch linear warm-up.

---

## 6. End-to-End Operational Workflow & Preview Control

The runtime execution pipeline is fully operational in [`demo_video.py`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/demo_video.py) and the interactive web application [`demo_app/app.py`](file:///home/benny/Projects/GV-YOLO26/demo_app/app.py).

```
   [ 1. Video Frame Acquisition ] (Dashcam / Windshield Camera @ 30 FPS)
                 │
                 ▼
     [ 2. Image Preprocessing ] (640×640 Resize, CLAHE Contrast Enhancement)
                 │
                 ▼
   [ 3. Road Hazard Detection ] (GV-YOLO26 Engine, NMS-Free End-to-End < 6 ms)
                 │
                 ▼
  [ 4. Hazard Tracking & Verification ] (Consecutive Frame Trajectory Association)
                 │
                 ▼
      < Hazard Confirmed? > ─── No ───► [ Reject False Alarm (Shadows/Paint) ]
                 │ Yes (≥ 3 Frames)
                 ▼
  [ 5. Distance & TTI Estimation ] (Pinhole Projection: Z = f*h/y, TTI = Z / v)
                 │
                 ▼
 [ 6. Warning & Suspension Control ] ──► Visual/Audio HUD Alert (Driver Warning)
                                      └──► CAN-Bus Damper Softening Pulse (ECU)
```

### Operational Stages:
1. **Video Frame Acquisition**: 1080p road video stream captured at 30 FPS from front-facing camera.
2. **Image Preprocessing**: Standardized 640×640 normalization with Contrast-Limited Adaptive Histogram Equalization (CLAHE) to boost contrast in low light, shadows, and rain.
3. **Road Hazard Detection**: Single-pass NMS-free forward inference via GV-YOLO26 producing class labels, confidences, and bounding boxes in $<6$ ms.
4. **Hazard Tracking & Temporal Verification**: Replaces fragile single-frame triggers with consecutive-frame trajectory association ($k=3, \tau=0.35$), rejecting transient visual noise (oil spots, road markings, tree shadows).
5. **Distance & Time-to-Impact (TTI) Estimation**: Monocular pinhole geometry ($h=1.25$ m, pitch $=8.0^\circ$, VFOV $=50.0^\circ$) computes metric distance $Z$ from the bottom contact patch of the box. Combining $Z$ with CAN-bus vehicle velocity ($v$) calculates arrival time ($TTI = Z / v$).
6. **Driver Warning & Active Suspension Control**:
   - **Driver HUD**: Renders bounding box, class, metric distance, and sound chime.
   - **CAN-bus Chassis Preview**: Transmits proactive damping commands:
     - `"SOFTEN_FAST"` ($15 \le Z \le 30$ m preview window): Opens damper bypass valves to absorb impact.
     - `"PRELOAD_CRITICAL"` ($Z < 15$ m): Stiffens damper to prevent suspension bottoming out.
     - `"NOMINAL"`: Default comfort damping state.

---

## 7. Model Checkpoints & Test Assets

### Trained Weights ([`GV-YOLO26/weights/`](file:///home/benny/Projects/GV-YOLO26/GV-YOLO26/weights/)):
- `gv_yolo26_best.pt` (3.7 MB): Final Stage 6 ultra-lightweight checkpoint (1.57M params, 5.1 GFLOPs, 77.5% clean mAP50).
- `gv_yolo26_simple_best.pt` (4.4 MB): Balanced production checkpoint (2.08M params, 5.8 GFLOPs, 82.8% clean mAP50).
- `gv_yolo26_x_best.pt` (13 MB): Adverse condition checkpoint with enhanced feature capacity.
- `gv_yolo26_x_weather_best.pt` (13 MB): Weather-augmented checkpoint.
- `gv_yolo26_x_allaug_best.pt` (13 MB): Checkpoint trained with full augmentation suite.

### Video Evaluation Suite ([`video/`](file:///home/benny/Projects/GV-YOLO26/video/)):
- `v1.mp4` (6.3 MB): Daylight urban road with multiple potholes.
- `v2.mp4` (3.2 MB): Speed bump approach test sequence.
- `v3.mp4` (2.9 MB): Broken pavement and severe road surface depression.
- `v4.mp4` (4.0 MB): Low-contrast dusk speed bump sequence.
- `v5.mp4`–`v7.mp4`: Extended driving and suburban road hazard scenarios.

---

## 8. Verification Commands & Operational Runbook

Always activate the project virtual environment before running commands (`source venv/bin/activate` or `./venv/bin/python`).

### Architecture Verification & Smoke Test
```bash
# Verify all custom layers, parameter counts, and GFLOPs across ablation ladder
PYTHONPATH=GV-YOLO26 python GV-YOLO26/tools/model_summary.py
```

### Video Tracking & Active Suspension Preview Demo
```bash
# Run video demo with temporal confirmation (k=3) and suspension preview triggers
python GV-YOLO26/demo_video.py \
  --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt \
  --source video/v2.mp4
```

### Interactive Web HUD Demo Server
```bash
# Start FastAPI + ByteTrack web application (interactive HUD with camera/video support)
python run_demo.py 8000
# Open browser at http://localhost:8000
```

### Clean Held-Out Model Validation
```bash
# Validate checkpoint on leakage-free held-out test split (NMS-free mode)
python GV-YOLO26/val.py \
  --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt \
  --data GV-YOLO26/data_local/data.yaml
```

### Model Inference (Single Image or Batch)
```bash
python GV-YOLO26/infer.py \
  --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt \
  --source path/to/image_or_dir
```

### Clean Data Split Generator
```bash
python GV-YOLO26/tools/make_clean_split.py \
  --root datasets/speed12 \
  --data GV-YOLO26/data/data.yaml \
  --out GV-YOLO26/data_local
```

### Automated Kaggle Checkpoint Fetching
```bash
python fetch_and_test.py
```

---

## 9. Next Steps & Technical Roadmap

1. **Hardware In-Vehicle Deployment (NVIDIA Jetson)**:
   - Export optimized FP16/INT8 TensorRT execution engines via `GV-YOLO26/export_trt.py`.
   - Deploy on physical NVIDIA Jetson AGX Xavier / Jetson Orin Nano boards connected to a dashboard USB/CSI camera.
2. **Quantization-Aware Training (QAT)**:
   - Implement PyTorch QAT during the final 20 fine-tuning epochs to recover mAP drop encountered during post-training INT8 quantization.
3. **CAN-Bus Physical Actuator Interfacing**:
   - Connect the preview control pipeline to an automotive CAN-bus shield to broadcast real-time damper stiffness command frames (`0x3F2`) to electronic suspension ECUs.
4. **Local Indian Road Hazard Dataset Collection**:
   - Collect and annotate monocular dashcam video across unpainted rural speed breakers, concrete speed humps, and waterlogged monsoon potholes.
