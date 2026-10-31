"""Distil GV-YOLO26 (N scale) from a trained GV-YOLO26-S teacher with BCKD + CWD (equation 4.5).

Example:
    python distill.py --teacher runs/teacher_s/weights/best.pt --name gv_yolo26_distilled
"""

import argparse

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from gv_modules import DistillTrainer, NWDDetectionLoss
from gv_modules.augment import weather_augmentations
from gv_modules.model import _KD
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="trained teacher checkpoint (best.pt)")
    p.add_argument("--student", default="models/gv-yolo26n.yaml", help="student YAML or a stage-6 checkpoint")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--weights", default="yolo26n.pt", help="COCO weights for a YAML student; ignored for a .pt")
    p.add_argument("--alpha", type=float, default=0.70)
    p.add_argument("--w-bckd", type=float, default=0.5)
    p.add_argument("--w-cwd", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (ignored by optimizer=auto)")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--distill-warmup-epochs", type=float, default=3.0, help="linear warmup epochs for distillation loss")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="gv_yolo26_distilled")
    return p.parse_args()


def main(opt):
    NWDDetectionLoss.alpha = opt.alpha
    _KD["weight_bckd"], _KD["weight_cwd"] = opt.w_bckd, opt.w_cwd
    _KD["alpha"] = opt.alpha
    DistillTrainer.teacher = opt.teacher
    DistillTrainer.distill_warmup_epochs = opt.distill_warmup_epochs
    model = YOLO(opt.student, task="detect")
    if opt.student.endswith((".yaml", ".yml")) and opt.weights:
        model.load(opt.weights)
    model.train(
        trainer=DistillTrainer,
        data=opt.data,
        imgsz=opt.imgsz,
        epochs=opt.epochs,
        patience=opt.patience,
        batch=opt.batch,
        seed=42,
        deterministic=True,
        optimizer=opt.optimizer,
        lr0=opt.lr0,
        warmup_epochs=opt.warmup_epochs,
        amp=True,
        close_mosaic=10,
        flipud=0.0,
        device=opt.device,
        workers=opt.workers,
        project=opt.project,
        name=opt.name,
        augmentations=weather_augmentations(),
    )


if __name__ == "__main__":
    main(parse_args())
