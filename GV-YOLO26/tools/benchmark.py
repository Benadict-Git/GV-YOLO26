"""Measure every trained model under one protocol and write results/benchmark.json (Tables 5.3 to 5.6).

For each checkpoint: accuracy on the val and test splits (one-to-one head, conf 0.001, batch 1), deployed-model
parameters and GFLOPs (fused, one-to-many branch removed), and end-to-end single-image latency on 100 test images
after 10 warm-up images. Selected checkpoints are also exported to TensorRT at FP32, FP16 and INT8 and measured the
same way (accuracy on test, latency, engine size).

Example:
    python tools/benchmark.py --models s1_yolo26n=runs/s1/weights/best.pt gv_yolo26=runs/s6/weights/best.pt \
        --trt s1_yolo26n gv_yolo26 --out results/benchmark.json
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gv_modules  # noqa: E402,F401  registers the GV-YOLO26 layers
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="name=path/to/best.pt")
    p.add_argument("--trt", nargs="*", default=[], help="names of models to also export to TensorRT")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--extra", nargs="*", default=[], help="name=data.yaml evaluated on that YAML's 'test' entry")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--timing-images", type=int, default=100)
    p.add_argument("--sweep", nargs="*", default=[], help="models to also evaluate at the --sizes input resolutions")
    p.add_argument("--sizes", nargs="*", type=int, default=[800, 960])
    p.add_argument("--out", default="results/benchmark.json")
    return p.parse_args()


def accuracy(model, data, split, imgsz):
    m = model.val(data=data, split=split, imgsz=imgsz, batch=1, conf=0.001, iou=0.7, device=0, plots=False,
                  nms=False, verbose=False)
    per = {m.names[c]: dict(P=float(m.box.p[k]), R=float(m.box.r[k]), ap50=float(m.box.ap50[k]), ap=float(m.box.ap[k]))
           for k, c in enumerate(m.box.ap_class_index)}
    return dict(P=float(m.box.mp), R=float(m.box.mr), map50=float(m.box.map50), map=float(m.box.map), per_class=per)


def latency(model, files, imgsz, warm=10):
    for f in files[:warm]:
        model.predict(f, imgsz=imgsz, device=0, nms=False, verbose=False)
    times = []
    for f in files[warm:]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, device=0, nms=False, verbose=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    ms = 1000 * float(np.mean(times))
    return dict(ms=ms, fps=1000 / ms)


def deployed_cost(model, imgsz):
    f = deepcopy(model.model).float().fuse(verbose=False)
    return dict(params_m=sum(p.numel() for p in f.parameters()) / 1e6, gflops=float(get_flops(f, imgsz)))


def main(opt):
    out = Path(opt.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else {}
    root = Path(check_det_dataset(opt.data)["test"])
    files = sorted(str(f) for f in root.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[: opt.timing_images + 10]

    for spec in opt.models:
        name, w = spec.split("=", 1)
        if name in results and "test" in results[name]:
            continue
        model = YOLO(w)
        r = dict(weights=w, **deployed_cost(model, opt.imgsz))
        r["val"] = accuracy(model, opt.data, "val", opt.imgsz)
        r["test"] = accuracy(model, opt.data, "test", opt.imgsz)
        for e in opt.extra:
            en, ey = e.split("=", 1)
            r[en] = accuracy(model, ey, "test", opt.imgsz)
        r["pytorch_fp32"] = latency(YOLO(w), files, opt.imgsz)
        results[name] = r
        out.write_text(json.dumps(results, indent=1))
        print(name, json.dumps(r))

    weights = dict(s.split("=", 1) for s in opt.models)
    for name in opt.sweep:
        for sz in opt.sizes:
            key = f"{name}@{sz}"
            if key in results:
                continue
            model = YOLO(weights[name])
            r = dict(gflops=float(get_flops(deepcopy(model.model).float().fuse(verbose=False), sz)))
            r["test"] = accuracy(model, opt.data, "test", sz)
            for e in opt.extra:
                en, ey = e.split("=", 1)
                r[en] = accuracy(model, ey, "test", sz)
            r["pytorch_fp32"] = latency(YOLO(weights[name]), files, sz)
            results[key] = r
            out.write_text(json.dumps(results, indent=1))
            print(key, json.dumps(r))
    for name in opt.trt:
        for prec in ["fp32", "fp16", "int8"]:
            key = f"{name}_trt_{prec}"
            if key in results:
                continue
            try:
                kw = dict(half=prec == "fp16", int8=prec == "int8")
                if kw["int8"]:
                    kw["data"] = opt.data
                engine = YOLO(weights[name]).export(format="engine", imgsz=opt.imgsz, device=0, batch=1, nms=False, **kw)
                trt = YOLO(engine, task="detect")
                r = dict(engine_mb=Path(engine).stat().st_size / 1e6, **latency(trt, files, opt.imgsz))
                r["test"] = accuracy(YOLO(engine, task="detect"), opt.data, "test", opt.imgsz)
                for e in opt.extra:
                    en, ey = e.split("=", 1)
                    r[en] = accuracy(YOLO(engine, task="detect"), ey, "test", opt.imgsz)
                Path(engine).rename(Path(engine).with_name(f"{Path(engine).stem}_{prec}.engine"))
            except Exception as e:  # keep the PyTorch results even if a TensorRT build fails
                r = dict(error=repr(e))
            results[key] = r
            out.write_text(json.dumps(results, indent=1))
            print(key, json.dumps(r))


if __name__ == "__main__":
    main(parse_args())
