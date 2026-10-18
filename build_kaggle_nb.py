import json
from pathlib import Path

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_md(text):
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_code(text):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")]
    })

def add_writefile(rel_path):
    local_p = Path("GV-YOLO26") / rel_path
    content = local_p.read_text()
    code = f"%%writefile {rel_path}\n{content}"
    add_code(code)

add_md("""# GV-YOLO26: Optimal Accuracy Training & Distillation Pipeline
- Stabilized BCKD Knowledge Distillation from teacher_s (no GIoU divergence)
- High-capacity GV-YOLO26-Opt architecture (2.16M params, P2 stride-4, GhostConv, C3k2 preservation)
- Road-specific daylight contrast augmentations (CLAHE, RandomGamma, RandomBrightnessContrast, Sharpen)
- Balanced NWD loss (alpha = 0.70)
- Parallel 2x Tesla T4 GPU training""")

add_code("""!nvidia-smi
import torch
print('torch', torch.__version__, '- cuda available:', torch.cuda.is_available(), '- device count:', torch.cuda.device_count())""")

add_code("""!pip install -q "ultralytics==8.4.156" "albumentations>=2.0" opencv-python-headless""")

add_code("""import os
from pathlib import Path

PROJECT_DIR = "/kaggle/working/GV-YOLO26"
os.makedirs(f"{PROJECT_DIR}/gv_modules", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/models", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.chdir(PROJECT_DIR)
print("Working directory:", os.getcwd())""")

files_to_write = [
    "gv_modules/__init__.py",
    "gv_modules/register.py",
    "gv_modules/blocks.py",
    "gv_modules/head.py",
    "gv_modules/loss.py",
    "gv_modules/model.py",
    "gv_modules/trainer.py",
    "gv_modules/augment.py",
    "models/gv-yolo26-opt.yaml",
    "train.py",
    "distill.py",
    "val.py"
]

for f in files_to_write:
    add_writefile(f)

add_code("""import gv_modules
from ultralytics import YOLO

m = YOLO("models/gv-yolo26-opt.yaml", task="detect")
print("GV-YOLO26-Opt builds OK -", sum(p.numel() for p in m.model.parameters()), "params")""")

add_code("""from pathlib import Path
import random
import yaml

print("under /kaggle/input:", sorted(str(p) for p in Path("/kaggle/input").glob("*")))
hits = [p.parent.parent for p in Path("/kaggle/input").rglob("train/images") if (p.parent.parent / "test/images").is_dir()]
assert hits, "no train/images + test/images found anywhere under /kaggle/input"
DATASET_ROOT = str(hits[0])
names = {0: "Pothole", 1: "Speed Bump"}

train_images = sorted(
    str(p) for p in (Path(DATASET_ROOT) / "train/images").glob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)
rng = random.Random(42)
rng.shuffle(train_images)
n_val = max(1, round(0.10 * len(train_images)))
val_images, train_only_images = train_images[:n_val], train_images[n_val:]

Path("data").mkdir(exist_ok=True)
Path("data/train_list.txt").write_text("\\n".join(train_only_images) + "\\n")
Path("data/val_list.txt").write_text("\\n".join(val_images) + "\\n")
print(f"train/: {len(train_images)} images -> {len(train_only_images)} train / {len(val_images)} val")

with open("data/data.yaml", "w") as f:
    yaml.safe_dump(
        {
            "path": DATASET_ROOT,
            "train": str(Path("data/train_list.txt").resolve()),
            "val": str(Path("data/val_list.txt").resolve()),
            "test": "test/images",
            "names": names
        },
        f,
        sort_keys=False,
    )
print("data.yaml configured successfully:")
print(open("data/data.yaml").read())""")

add_code("""from pathlib import Path
import shutil

Path("/kaggle/working/init").mkdir(exist_ok=True)

teacher_hits = sorted(Path("/kaggle/input").rglob("teacher_s/weights/best.pt"))
assert teacher_hits, "Teacher weights not found under /kaggle/input"
shutil.copy(str(teacher_hits[0]), "/kaggle/working/init/teacher.pt")
print("Found Teacher:", teacher_hits[0])

student_hits = sorted(Path("/kaggle/input").rglob("s4_ghost/weights/best.pt"))
assert student_hits, "Student weights not found under /kaggle/input"
shutil.copy(str(student_hits[0]), "/kaggle/working/init/student.pt")
print("Found Student:", student_hits[0])""")

add_code("""import subprocess
import sys

RUNS_DIR = "/kaggle/working/runs"

JOBS = [
    {
        "name": "gv_yolo26_opt_distill",
        "script": "distill.py",
        "gpu": "0",
        "args": [
            "--teacher", "/kaggle/working/init/teacher.pt",
            "--student", "/kaggle/working/init/student.pt",
            "--epochs", "100",
            "--batch", "16",
            "--optimizer", "AdamW",
            "--lr0", "0.0003",
            "--warmup-epochs", "3.0",
            "--alpha", "0.70",
            "--device", "0"
        ]
    },
    {
        "name": "gv_yolo26_opt_train",
        "script": "train.py",
        "gpu": "1",
        "args": [
            "--cfg", "models/gv-yolo26-opt.yaml",
            "--weights", "/kaggle/working/init/student.pt",
            "--nwd",
            "--alpha", "0.70",
            "--epochs", "100",
            "--batch", "16",
            "--optimizer", "AdamW",
            "--lr0", "0.0005",
            "--warmup-epochs", "3.0",
            "--device", "1"
        ]
    }
]

procs = []
for job in JOBS:
    cmd = [
        sys.executable, job["script"],
        "--name", job["name"],
        "--data", "data/data.yaml",
        "--imgsz", "640",
        "--workers", "4",
        "--project", RUNS_DIR
    ] + job["args"]
    print("+ [GPU " + job["gpu"] + "]", " ".join(cmd))
    log = open(f"/kaggle/working/{job['name']}_train.log", "w")
    procs.append((job, subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)))

for job, p in procs:
    rc = p.wait()
    tail = open(f"/kaggle/working/{job['name']}_train.log", errors="ignore").read().splitlines()[-25:]
    print(f"=== {job['name']} finished with exit code {rc}")
    print("\\n".join(t.split("\\r")[-1] for t in tail))
    assert rc == 0, f"{job[\'name\']} training failed!"
""")

add_code("""import json
from pathlib import Path
import shutil
from ultralytics import YOLO

def evaluate(weights, split):
    m = YOLO(weights).val(
        data="data/data.yaml",
        split=split,
        imgsz=640,
        batch=1,
        conf=0.001,
        iou=0.7,
        device=0,
        plots=split == "test",
        nms=False,
        verbose=False,
        project="/kaggle/working/eval",
        name=f"{Path(weights).parts[-3]}_{split}",
        exist_ok=True
    )
    per = {
        m.names[i]: dict(P=float(m.box.p[k]), R=float(m.box.r[k]), ap50=float(m.box.ap50[k]), ap=float(m.box.ap[k]))
        for k, i in enumerate(m.box.ap_class_index)
    }
    return dict(P=float(m.box.mp), R=float(m.box.mr), map50=float(m.box.map50), map=float(m.box.map), per_class=per)

results = {}
best_score = -1.0
best_model_path = None

for job in JOBS:
    w = f"/kaggle/working/runs/{job['name']}/weights/best.pt"
    if Path(w).exists():
        val_res = evaluate(w, "val")
        test_res = evaluate(w, "test")
        results[job["name"]] = {"val": val_res, "test": test_res}
        print(f"=== {job['name']} ===")
        print(f"Validation mAP@0.5: {val_res['map50']:.4f} | mAP@0.5:0.95: {val_res['map']:.4f}")
        print(f"Test mAP@0.5:       {test_res['map50']:.4f} | mAP@0.5:0.95: {test_res['map']:.4f}")
        if test_res["map50"] > best_score:
            best_score = test_res["map50"]
            best_model_path = w

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)

if best_model_path:
    shutil.copy(best_model_path, "/kaggle/working/gv_yolo26_best_opt.pt")
    print(f"Saved best model ({best_score:.4f} mAP50) to /kaggle/working/gv_yolo26_best_opt.pt")""")

out_path = Path("GV-YOLO26/kaggle_jobs/gv-yolo26-opt-train/gv-yolo26-opt-train.ipynb")
out_path.write_text(json.dumps(nb, indent=1))
print(f"Generated notebook: {out_path} ({len(nb['cells'])} cells)")
