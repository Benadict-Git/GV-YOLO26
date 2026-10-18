"""Train one GV-YOLO26 design stage (Table 4.5) with the configuration of Table 4.6.

Examples:
    python train.py --cfg yolo26n.yaml --name s1_yolo26n                                  # stage 1
    python train.py --cfg models/ablation/gv-yolo26n-p2.yaml --name s2_p2                 # stage 2
    python train.py --cfg models/ablation/gv-yolo26n-p2-lgdh.yaml --name s3_lgdh          # stage 3
    python train.py --cfg models/ablation/gv-yolo26n-p2-lgdh-ghost.yaml --name s4_ghost   # stage 4
    python train.py --cfg models/gv-yolo26n.yaml --name s5_vov                            # stage 5
    python train.py --cfg models/gv-yolo26n.yaml --nwd --name s6_gv_yolo26                # stage 6 (final)
    python train.py --cfg models/gv-yolo26s.yaml --nwd --name teacher_s --weights yolo26s.pt   # teacher
"""

import argparse

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from gv_modules import GVTrainer, NWDDetectionLoss, NWDTrainer
from gv_modules.augment import weather_augmentations
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="models/gv-yolo26n.yaml", help="model YAML (scale letter after 'yolo26') or a .pt to fine-tune")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--weights", default="yolo26n.pt", help="COCO checkpoint for layers whose shapes match; '' = none")
    p.add_argument("--nwd", action="store_true", help="use the IoU + NWD box loss (stage 6)")
    p.add_argument("--alpha", type=float, default=0.70, help="IoU weight in the box loss (equation 4.4)")
    p.add_argument("--no-weather", action="store_true", help="disable rain/snow/blur/colour augmentation")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.01, help="initial learning rate (ignored by optimizer=auto)")
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="gv_yolo26")
    return p.parse_args()


def main(opt):
    NWDDetectionLoss.alpha = opt.alpha
    model = YOLO(opt.cfg, task="detect")
    if opt.weights:
        model.load(opt.weights)  # transfers only the layers whose names and shapes match
    extra = {} if opt.no_weather else {"augmentations": weather_augmentations()}
    model.train(
        trainer=NWDTrainer if opt.nwd else GVTrainer,
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
        **extra,
    )


if __name__ == "__main__":
    main(parse_args())
