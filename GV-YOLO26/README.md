# GV-YOLO26: Lightweight Real-Time Detection of Speed Bumps and Potholes

YOLO26n rebuilt with GhostConv + VoVGSCSPC (backbone/neck), a P2 detection level, a Lightweight Grouped Detection
Head (LGDH) with NMS-free one-to-one prediction, and an IoU + NWD box loss. Two classes: `Pothole`, `Speed Bump`.

## Setup

```bash
python -m pip install -r requirements.txt   # ultralytics==8.4.156 is required exactly (layer registration)
```

## In-video detection demo

```bash
# annotate every clip in ../video, write H.264 videos + summary.json to results/video
python demo_video.py --weights weights/gv_yolo26_best.pt --source ../video --out results/video

# live window while processing (press q to quit); on Wayland, prefix with QT_QPA_PLATFORM=xcb if no window appears
python demo_video.py --weights weights/gv_yolo26_best.pt --source ../video/v2.mp4 --show
```

Detections are linked across frames with ByteTrack; a hazard is drawn and the warning banner raised only after its
track has been seen in 3 frames (`--confirm`) at confidence >= 0.35 (`--conf`). `--confirm 1` disables the filter.

## Images

```bash
python infer.py --weights weights/gv_yolo26_best.pt --source path/to/images_or_video
```

## Training and evaluation (Kaggle, 2 x T4)

| Step | Script |
|---|---|
| One design stage | `python train.py --cfg models/gv-yolo26n.yaml --nwd --name s6_gv_yolo26` (stages in `train.py`) |
| Distillation | `python distill.py --teacher runs/teacher_s/weights/best.pt` |
| Leakage-free held-out set + dataset stats | `python tools/make_clean_split.py --root <speed12 root>` |
| All metrics, latency, TensorRT FP32/FP16/INT8 | `python tools/benchmark.py --models name=best.pt ... --trt name ...` |

`Kaggle_Train_GV_YOLO26.ipynb` trains on Kaggle end to end; `kaggle_jobs/` holds the notebooks used for the ablation,
distillation and benchmark runs reported in the project report.

**Dataset note:** the speed12 `test/` and `valid/` folders contain augmented copies of `train/` photos and other frames
of the same videos. Report results on the leakage-free held-out set built by `tools/make_clean_split.py`.
