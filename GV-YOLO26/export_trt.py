"""Export a trained model to a TensorRT engine and time it (Table 5.5). Run on the Jetson itself.

Examples:
    python export_trt.py --weights runs/s6_gv_yolo26/weights/best.pt --precision fp16
    python export_trt.py --weights runs/s6_gv_yolo26/weights/best.pt --precision int8 --data data/data.yaml
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    p.add_argument("--data", default="data/data.yaml", help="calibration images for INT8")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--runs", type=int, default=500)
    p.add_argument("--nms", action="store_true", help="export the one-to-many head for external NMS instead")
    return p.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    kw = {"half": opt.precision == "fp16", "int8": opt.precision == "int8", "nms": None if opt.nms else False}
    if kw["int8"]:
        kw["data"] = opt.data
    engine = model.export(format="engine", imgsz=opt.imgsz, device=0, batch=1, **kw)

    trt = YOLO(engine, task="detect")
    x = np.random.randint(0, 255, (opt.imgsz, opt.imgsz, 3), dtype=np.uint8)
    for _ in range(50):  # warm-up
        trt.predict(x, imgsz=opt.imgsz, nms=kw["nms"], verbose=False)
    times = []
    for _ in range(opt.runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        trt.predict(x, imgsz=opt.imgsz, nms=kw["nms"], verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(times))
    size = Path(engine).stat().st_size / 1e6
    print(f"{opt.precision.upper()}: {ms:.2f} ms  {1000 / ms:.1f} FPS  engine {size:.2f} MB")


if __name__ == "__main__":
    main(parse_args())
