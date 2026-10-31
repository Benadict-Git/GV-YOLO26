# AGENTS.md

GV-YOLO26 is an ultra-lightweight, NMS-free road hazard detector (`Pothole`, `Speed Bump`) built on YOLO26n for active vehicle suspension preview control. It incorporates GhostConv, VoVGSCSPC, a high-resolution P2 branch (160x160), Lightweight Grouped Detection Head (LGDH), and Normalized Wasserstein Distance (NWD) loss.

## Commands

Always use the project virtual environment (`source venv/bin/activate` or `./venv/bin/python`).
*Note: If invoking `./venv/bin/pip` directly fails due to relocated shebangs, run via `python -m pip`.*

### Verification & Smoke Test
- Verify layer registration, model compilation, FLOPs, and params across all ablation stages:
  ```bash
  PYTHONPATH=GV-YOLO26 python GV-YOLO26/tools/model_summary.py
  ```

### Inference & Demo Server
- Start mobile HUD demo server (FastAPI + ByteTrack):
  ```bash
  python run_demo.py [port]  # default port 8000
  ```
- Video tracking demo with temporal hazard confirmation:
  ```bash
  python GV-YOLO26/demo_video.py --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt --source video/v2.mp4
  ```
- Fetch completed Kaggle training outputs and run video suite:
  ```bash
  python fetch_and_test.py
  ```
- Single image or batch directory inference:
  ```bash
  python GV-YOLO26/infer.py --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt --source path/to/images_or_video
  ```

### Evaluation & Benchmarking
- Validate on held-out dataset & measure single-image latency:
  ```bash
  python GV-YOLO26/val.py --weights GV-YOLO26/weights/gv_yolo26_simple_best.pt --data GV-YOLO26/data_local/data.yaml
  ```
- Comprehensive metric benchmark (PyTorch & TensorRT):
  ```bash
  python GV-YOLO26/tools/benchmark.py --models gv_yolo26=GV-YOLO26/weights/gv_yolo26_simple_best.pt --out results/benchmark.json
  ```

### Training & Dataset Preparation
- Train a stage with NWD loss:
  ```bash
  python GV-YOLO26/train.py --cfg GV-YOLO26/models/gv-yolo26.yaml --nwd --name s6_gv_yolo26
  ```
- Build leakage-free held-out dataset split:
  ```bash
  python GV-YOLO26/tools/make_clean_split.py --root datasets/speed12 --data GV-YOLO26/data/data.yaml --out GV-YOLO26/data_local
  ```

### Report Generation
- Rebuild academic report (.docx) from measured experiment JSONs:
  ```bash
  python report_build/build_report.py report_build/inputs report_build/GV_YOLO26_Project_Report.docx
  ```

## Architecture

- **`GV-YOLO26/gv_modules/`**: Core custom architectural components.
  - `register.py`: Runtime monkey-patch of Ultralytics `parse_model` to register `GSConv`, `VoVGSCSPC`, and `LGDH`. Triggered automatically upon `import gv_modules`.
  - `blocks.py`: `GSConv`, `GSBottleneckC`, `VoVGSCSPC` for cheap convolution and spatial feature mixing.
  - `head.py`: `LGDH` (Lightweight Grouped Detection Head) using grouped convolutions (`channels_per_group=8`) to control P2 FLOP complexity.
  - `loss.py`: `NWDDetectionLoss` combining CIoU with Normalized Wasserstein Distance to stabilize regression on tiny bounding boxes.
  - `augment.py`: Albumentations weather simulation pipeline (rain, snow, motion blur, lighting shifts).
  - `trainer.py`: `GVTrainer`, `NWDTrainer`, and `DistillTrainer`.
- **`GV-YOLO26/models/`**: Architecture YAML configurations across the 6-stage ablation ladder (`gv-yolo26.yaml`, `ablation/gv-yolo26-p2.yaml`, etc.).
- **`demo_app/`**: Real-time HUD web app (`app.py`, `sbp_modules.py`) allowing interactive camera streaming and dual-model comparison between GV-YOLO26 and SBP-YOLO.
- **`report_build/`**: Automated script suite (`build_report.py`, `docxlib.py`, `narrative.py`) injecting empirical benchmark JSONs into the formal academic project report.

## Conventions

- **Ultralytics version is strictly pinned**: Must stay `ultralytics==8.4.156`. `gv_modules/register.py` relies on exact string anchors in `tasks.parse_model`. Do not upgrade Ultralytics without updating those patch anchors.
- **Always import custom modules before loading models**: Any script loading GV-YOLO26 weights or YAMLs must execute `import gv_modules` first. For SBP-YOLO models, execute `import sbp_modules; sbp_modules.register_sbp_modules()`.
- **NMS-Free inference**: YOLO26 uses one-to-one end-to-end heads. Default inference and validation must pass `nms=False`. Only pass `--nms` if explicitly testing the one-to-many auxiliary head.
- **Data leakage prevention**: The raw `speed12` test/valid sets contain augmented duplicates and consecutive video frames shared with `train/`. Benchmark results must always be evaluated on the leakage-free split created by `tools/make_clean_split.py` (`clean_holdout.txt` / `data_clean.yaml`).
- **External clones**: `ultralytics-distill/` and `ultralytics-yolo11/` are gitignored reference repositories; do not commit changes there.
