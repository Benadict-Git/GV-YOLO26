"""Train the seven stock baselines under one fixed protocol (Table 4.7)."""

from ultralytics import RTDETR, YOLO

MODELS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv9t": "yolov9t.pt",
    "YOLOv10n": "yolov10n.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO12n": "yolo12n.pt",
    "YOLO26n": "yolo26n.pt",
    "RT-DETR-L": "rtdetr-l.pt",
}
COMMON = dict(
    data="data/data.yaml",
    imgsz=640,
    epochs=50,
    patience=15,
    seed=42,
    deterministic=True,
    optimizer="auto",
    amp=True,
    close_mosaic=10,
    flipud=0.0,
    project="runs/baselines",
)

if __name__ == "__main__":
    for name, ckpt in MODELS.items():
        is_detr = name.startswith("RT-DETR")
        model = RTDETR(ckpt) if is_detr else YOLO(ckpt)
        model.train(batch=8 if is_detr else 16, name=name, **COMMON)
