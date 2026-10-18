"""Report figures 4.8-4.10: detections on held-out images and frames from the in-video demo.

Example:
    python tools/make_figures.py --weights weights/gv_yolo26_best.pt --images <dir of held-out images> \
        --videos results/video --out results/figures
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gv_modules  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402

FIG48 = [("1962_jpg.rf.1307d362f3ac8d37e4148c4116439770.jpg", "(a) distant unpainted bump"),
         ("images-19-_jpg.rf.79ff37f5cfe5d42943c71a15c8de394b.jpg", "(b) painted bump")]
FIG49 = [("V0F_HY_3425_20210104_140059_E_CH2_Busan_Sun_Mainroad_Day_42498_png.rf.*.jpg", "(a) potholes on an urban road"),
         ("add_motion_blur_vlcsnap-2020-04-12-02h08m18s958_jpg.rf.*.jpg", "(b) a distant pothole")]
FIG410 = [("v1_gv.mp4", 30, "(a) day"), ("v1_gv.mp4", 230, "(b) rain"), ("v2_gv.mp4", 80, "(c) night, bump"),
          ("v3_gv.mp4", 230, "(d) night, potholes")]


def panel(img, label, w=640):
    img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    bar = np.full((48, w, 3), 255, np.uint8)
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(bar, label, ((w - tw) // 2, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([img, bar])


def row(panels, gap=16):
    h = max(p.shape[0] for p in panels)
    padded = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for p in panels]
    sep = np.full((h, gap, 3), 255, np.uint8)
    out = padded[0]
    for p in padded[1:]:
        out = np.hstack([out, sep, p])
    return out


def find(images, pattern):
    hits = sorted(Path(images).rglob(pattern))
    assert hits, pattern
    return hits[0]


def main(opt):
    model = YOLO(opt.weights)
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in [("fig4_8.png", FIG48), ("fig4_9.png", FIG49)]:
        panels = []
        for pattern, label in spec:
            r = model.predict(str(find(opt.images, pattern)), imgsz=640, conf=0.25, nms=False, verbose=False)[0]
            panels.append(panel(r.plot(line_width=3, font_size=18), label))
            print(name, label, [f"{r.names[int(b.cls)]} {float(b.conf):.2f}" for b in r.boxes])
        cv2.imwrite(str(out / name), row(panels))
    frames = []
    for clip, idx, label in FIG410:
        cap = cv2.VideoCapture(str(Path(opt.videos) / clip))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        assert ok, (clip, idx)
        frames.append(panel(f, label))
    grid = np.vstack([row(frames[:2]), np.full((16, row(frames[:2]).shape[1], 3), 255, np.uint8), row(frames[2:])])
    cv2.imwrite(str(out / "fig4_10.png"), grid)
    print("written", sorted(p.name for p in out.glob("fig4_*.png")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--videos", default="results/video")
    p.add_argument("--out", default="results/figures")
    main(p.parse_args())
