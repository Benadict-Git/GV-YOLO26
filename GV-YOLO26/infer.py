"""Run a trained model on images, a folder or a video and save annotated outputs (Figs. 4.8 and 4.9).

Examples:
    python infer.py --weights runs/s6_gv_yolo26/weights/best.pt --source samples/
    python infer.py --weights runs/s6_gv_yolo26/weights/best.pt --source clips/road.mp4
"""

import argparse
from pathlib import Path

import cv2

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from ultralytics import YOLO

VIDEO = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", default="results")
    p.add_argument("--nms", action="store_true", help="use the one-to-many head + NMS instead of the NMS-free head")
    return p.parse_args()


def main(opt):
    model = YOLO(opt.weights)
    nms = None if opt.nms else False
    src = Path(opt.source)
    if src.suffix.lower() in VIDEO:
        Path(opt.out).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dst = Path(opt.out) / f"{src.stem}_gv.mp4"
        out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            r = model.predict(frame, imgsz=opt.imgsz, conf=opt.conf, nms=nms, verbose=False)[0]
            out.write(r.plot())
        cap.release()
        out.release()
        print(f"saved {dst}")
    else:
        results = model.predict(str(src), imgsz=opt.imgsz, conf=opt.conf, nms=nms, save=True, project=opt.out,
                                name="gv_images", exist_ok=True)
        for r in results:
            dets = [f"{r.names[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes]
            more = f" (+{len(dets) - 10} more)" if len(dets) > 10 else ""
            print(f"{Path(r.path).name}: {', '.join(dets[:10]) or 'no detections'}{more}")


if __name__ == "__main__":
    main(parse_args())
