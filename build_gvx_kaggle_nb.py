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

# 1. Header
add_md("""# GV-YOLO26-X: Enhanced Real-Time Road Hazard Detector Training
- **Model**: `GV-YOLO26-X` (`models/gv-yolo26-x.yaml`)
- **Backbone**: SBP-YOLO efficient backbone stages (GhostConv + VoVGSCSPC)
- **Small Hazard Sensitivity**: Lossless Space-to-Depth (`SPDConv`) P2->P3 injection for sub-32px pothole detection
- **Wide Hazard Capacity**: Preserved `C3k2` at Stage 5 (P5) for wide road speed bumps
- **Head**: YOLO26 NMS-free end-to-end Detect head with standard 3-scale detection (8,400 anchors)
- **Loss**: Normalized Wasserstein Distance (`NWD`) box regression loss
- **Augmentation**: Adverse weather + dusk/low-contrast/shadow/faded-marking edge-case pipeline (`--all-aug`)
- **Dataset**: `elangovan001/speed12` with rigorous leakage-free clean split audit
""")

# 2. GPU Check
add_code("""!nvidia-smi
import torch
print('PyTorch version:', torch.__version__, '| CUDA available:', torch.cuda.is_available(), '| Device count:', torch.cuda.device_count())
""")

# 3. Dependencies
add_code("""!pip install -q "ultralytics==8.4.156" "albumentations>=2.0" opencv-python-headless""")

# 4. Project Workspace
add_code("""import os
from pathlib import Path

PROJECT_DIR = "/kaggle/working/GV-YOLO26"
os.makedirs(f"{PROJECT_DIR}/gv_modules", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/models", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/tools", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.chdir(PROJECT_DIR)
print("Working directory set to:", os.getcwd())
""")

# 5. Write source modules
files_to_write = [
    "gv_modules/__init__.py",
    "gv_modules/register.py",
    "gv_modules/blocks.py",
    "gv_modules/head.py",
    "gv_modules/loss.py",
    "gv_modules/model.py",
    "gv_modules/suspension_preview.py",
    "gv_modules/trainer.py",
    "gv_modules/augment.py",
    "models/gv-yolo26-x.yaml",
    "models/gv-yolo26-p2p3.yaml",
    "tools/make_clean_split.py",
    "train.py",
    "val.py"
]

for f in files_to_write:
    add_writefile(f)

# 6. Verify Architecture Initialization
add_code("""import gv_modules
from ultralytics import YOLO

m = YOLO("models/gv-yolo26-x.yaml", task="detect")
total_params = sum(p.numel() for p in m.model.parameters())
print(f"GV-YOLO26-X initialized successfully! Total parameters: {total_params:,}")
""")

# 7. Dataset Setup & Leakage-Free Clean Split
add_code("""from pathlib import Path
import random
import yaml
import subprocess

print("Scanning /kaggle/input for speed12 dataset...")
hits = [p.parent.parent for p in Path("/kaggle/input").rglob("train/images") if (p.parent.parent / "test/images").is_dir()]
assert hits, "No train/images + test/images found anywhere under /kaggle/input - please check attached dataset"
DATASET_ROOT = str(hits[0])
names = {0: "Pothole", 1: "Speed Bump"}
print("Found dataset root at:", DATASET_ROOT)

# 1. Carve val split (10%) from train/
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

# Standard data.yaml (pointing to untouched test/images)
data_cfg = {
    "path": DATASET_ROOT,
    "train": str(Path("data/train_list.txt").resolve()),
    "val": str(Path("data/val_list.txt").resolve()),
    "test": "test/images",
    "names": names
}
with open("data/data.yaml", "w") as f:
    yaml.safe_dump(data_cfg, f, sort_keys=False)

# 2. Build leakage-free clean holdout split
cmd_split = [
    "python", "tools/make_clean_split.py",
    "--root", DATASET_ROOT,
    "--data", "data/data.yaml",
    "--out", "data"
]
print("Running make_clean_split.py to isolate uncontaminated holdout...")
res = subprocess.run(cmd_split, capture_output=True, text=True)
print(res.stdout)
if res.returncode != 0:
    print("Clean split warning:", res.stderr)

print("data/data.yaml:")
print(open("data/data.yaml").read())
if Path("data/data_clean.yaml").exists():
    print("data/data_clean.yaml:")
    print(open("data/data_clean.yaml").read())
""")

# 8. Pre-download font and pretrained weights
add_code("""from ultralytics import YOLO
from ultralytics.utils.checks import check_font

check_font("Arial.ttf")
YOLO("yolo26n.pt")
print("Pre-downloaded Arial.ttf and yolo26n.pt successfully")
""")

# 9. Training Execution
add_code("""import subprocess
import sys
import torch
from pathlib import Path

RUNS_DIR = "/kaggle/working/runs"
n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} available GPU(s)")

if n_gpus >= 2:
    JOBS = [
        {
            "name": "gv_yolo26_x_allaug",
            "gpu": "0",
            "args": [
                "--cfg", "models/gv-yolo26-x.yaml",
                "--weights", "yolo26n.pt",
                "--nwd",
                "--all-aug",
                "--epochs", "150",
                "--patience", "30",
                "--batch", "16",
                "--device", "0"
            ]
        },
        {
            "name": "gv_yolo26_x_weather",
            "gpu": "1",
            "args": [
                "--cfg", "models/gv-yolo26-x.yaml",
                "--weights", "yolo26n.pt",
                "--nwd",
                "--epochs", "150",
                "--patience", "30",
                "--batch", "16",
                "--device", "1"
            ]
        }
    ]
else:
    dev = "0" if n_gpus == 1 else "cpu"
    JOBS = [
        {
            "name": "gv_yolo26_x_allaug",
            "gpu": dev,
            "args": [
                "--cfg", "models/gv-yolo26-x.yaml",
                "--weights", "yolo26n.pt",
                "--nwd",
                "--all-aug",
                "--epochs", "150",
                "--patience", "30",
                "--batch", "16",
                "--device", dev
            ]
        }
    ]

data_file = "data/data_clean.yaml" if Path("data/data_clean.yaml").exists() else "data/data.yaml"
print(f"Using training data configuration: {data_file}")

procs = []
for job in JOBS:
    cmd = [
        sys.executable, "train.py",
        "--name", job["name"],
        "--data", data_file,
        "--imgsz", "640",
        "--workers", "4",
        "--project", RUNS_DIR
    ] + job["args"]
    print(f"+ [GPU {job['gpu']}]", " ".join(cmd))
    log = open(f"/kaggle/working/{job['name']}_train.log", "w")
    procs.append((job, subprocess.Popen(cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)))

for job, p in procs:
    rc = p.wait()
    tail = open(f"/kaggle/working/{job['name']}_train.log", errors="ignore").read().splitlines()[-35:]
    print(f"=== {job['name']} finished with exit code {rc} ===")
    print("\\n".join(t.split("\\r")[-1] for t in tail))
    assert rc == 0, f"{job['name']} training failed!"
""")

# 10. Evaluation & Checkpoint Export
add_code("""import json
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

def evaluate(weights, data_path, split, name_suffix=""):
    dev = 0 if torch.cuda.is_available() else "cpu"
    m = YOLO(weights).val(
        data=data_path,
        split=split,
        imgsz=640,
        batch=1,
        conf=0.001,
        iou=0.7,
        device=dev,
        plots=(split == "test"),
        nms=False,
        verbose=False,
        project="/kaggle/working/eval",
        name=f"{Path(weights).parts[-3]}_{split}{name_suffix}",
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
        job_results = {}
        # 1. Validation set
        val_res = evaluate(w, "data/data.yaml", "val")
        job_results["val"] = val_res
        
        # 2. Clean holdout set
        if Path("data/data_clean.yaml").exists():
            clean_res = evaluate(w, "data/data_clean.yaml", "test", name_suffix="_clean")
            job_results["clean_holdout"] = clean_res
            print(f"[{job['name']}] Clean Holdout mAP@0.5: {clean_res['map50']:.4f} | mAP@0.5:0.95: {clean_res['map']:.4f}")
        
        # 3. Raw test set
        raw_res = evaluate(w, "data/data.yaml", "test", name_suffix="_raw")
        job_results["raw_test"] = raw_res
        print(f"[{job['name']}] Raw Test mAP@0.5:       {raw_res['map50']:.4f} | mAP@0.5:0.95: {raw_res['map']:.4f}")
        
        results[job["name"]] = job_results
        
        score = job_results.get("clean_holdout", raw_res)["map50"]
        if score > best_score:
            best_score = score
            best_model_path = w

with open("/kaggle/working/results.json", "w") as f:
    json.dump(results, f, indent=2)

if best_model_path:
    target_pt = "/kaggle/working/gv_yolo26_x_best.pt"
    shutil.copy(best_model_path, target_pt)
    print(f"\\nSaved top model ({best_score:.4f} mAP50) to {target_pt}")

print("\\nEvaluation complete. Results saved to /kaggle/working/results.json")
""")

# Output paths
job_dir = Path("GV-YOLO26/kaggle_jobs/gv-yolo26-x-train")
job_dir.mkdir(parents=True, exist_ok=True)

nb_path = job_dir / "gv-yolo26-x-train.ipynb"
nb_path.write_text(json.dumps(nb, indent=1))
print(f"Generated notebook: {nb_path} ({len(nb['cells'])} cells)")

# Write kernel metadata
metadata = {
    "id": "benadictinfanta/gv-yolo26-x-train",
    "title": "GV-YOLO26-X Train",
    "code_file": "gv-yolo26-x-train.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "elangovan001/speed12"
    ],
    "competition_sources": [],
    "kernel_sources": []
}

meta_path = job_dir / "kernel-metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"Generated kernel metadata: {meta_path}")
