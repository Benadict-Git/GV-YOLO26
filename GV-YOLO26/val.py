"""Evaluate a trained model once on the test split and time single-image inference (Tables 5.1 and 5.3).

Example:
    python val.py --weights runs/s6_gv_yolo26/weights/best.pt
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--timing-images", type=int, default=100)
    p.add_argument("--nms", action="store_true", help="use the one-to-many head + NMS instead of the NMS-free head")
    return p.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    nms = None if opt.nms else False  # False selects the one-to-one (NMS-free) head of end-to-end models
    m = model.val(data=opt.data, split=opt.split, imgsz=opt.imgsz, batch=1, conf=0.001, iou=0.7,
                  device=opt.device, plots=True, nms=nms)
    print(f"P {m.box.mp:.3f}  R {m.box.mr:.3f}  mAP@0.5 {m.box.map50:.3f}  mAP@0.5:0.95 {m.box.map:.3f}")
    for i, name in m.names.items():
        print(f"  {name:12s} AP@0.5 {m.box.ap50[i]:.3f}  AP@0.5:0.95 {m.box.ap[i]:.3f}")

    # end-to-end single-image latency: up to 10 warm-up images, then the mean over the next N
    root = Path(check_det_dataset(opt.data)[opt.split])
    files = sorted(str(f) for f in root.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    warm = min(10, len(files) // 2)
    files = files[: opt.timing_images + warm]
    times = []
    for k, f in enumerate(files):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(f, imgsz=opt.imgsz, device=opt.device, nms=nms, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if k >= warm:
            times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(times))
    print(f"latency {ms:.1f} ms  ->  {1000 / ms:.1f} FPS over {len(times)} images")


if __name__ == "__main__":
    main(parse_args())
