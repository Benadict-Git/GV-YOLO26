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

add_md("""# GV-YOLO26-Simple: Pure Backbone-Modified YOLO26 Training
- Model architecture: Stock YOLO26 with only backbone stages P4/P5 replaced with GhostConv + VoVGSCSPC
- Head & Neck: 100% Stock YOLO26 (P3, P4, P5 end2end Detect head)
- Dataset: elangovan001/speed12 (Pothole & Speed Bump)
- Pretrained weights transfer: Official yolo26n.pt
""")

add_code("""!nvidia-smi
import torch
print('torch:', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device count:', torch.cuda.device_count())""")

pip_cmd = "!pip " + "install -q ultralytics==8.4.156 opencv-python-headless"
add_code(pip_cmd)

add_code("""import os
from pathlib import Path

PROJECT_DIR = "/kaggle/working/GV-YOLO26"
os.makedirs(f"{PROJECT_DIR}/gv_modules", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/models", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/data", exist_ok=True)
os.chdir(PROJECT_DIR)
print("Working directory:", os.getcwd())""")

add_code('''%%writefile gv_modules/blocks.py
"""Cheap convolution blocks used in the GV-YOLO26 backbone."""
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv, DWConv

class GSConv(nn.Module):
    """Dense half from a standard convolution, cheap half from a depthwise convolution, then a channel shuffle."""
    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.shape
        return x2.view(b, 2, n // 2, h, w).transpose(1, 2).reshape(b, n, h, w)

class GSBottleneckC(nn.Module):
    """Two GSConv layers with a cheap depthwise shortcut."""
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GSConv(c1, c_, 1, 1), GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = DWConv(c1, c2, k, s, act=False)

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class VoVGSCSPC(nn.Module):
    """Cross-stage partial block with GSConv bottlenecks and one-shot aggregation."""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneckC(c_, c_) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        return self.cv3(torch.cat((self.cv2(x), self.gsb(self.cv1(x))), 1))
''')

add_code('''%%writefile gv_modules/register.py
"""Make VoVGSCSPC and GSConv usable from Ultralytics YAML model files."""
import inspect
import ultralytics
import ultralytics.nn.tasks as tasks
from .blocks import GSConv, VoVGSCSPC

TESTED_VERSION = "8.4.156"

def _patch_source(src):
    base_end = "            A2C2f,\\n        }\\n    )\\n    repeat_modules"
    repeat_end = "            A2C2f,\\n        }\\n    )\\n    for i, (f, n, m, args)"
    for anchor in (base_end, repeat_end):
        if src.count(anchor) != 1:
            raise RuntimeError(
                f"gv_modules: Ultralytics {ultralytics.__version__} is not supported "
                f"(parse_model layout changed; tested with {TESTED_VERSION})."
            )
    src = src.replace(base_end, base_end.replace("A2C2f,\\n", "A2C2f,\\n            GSConv,\\n            VoVGSCSPC,\\n"))
    src = src.replace(repeat_end, repeat_end.replace("A2C2f,\\n", "A2C2f,\\n            VoVGSCSPC,\\n"))
    return src

def register():
    if getattr(tasks, "_gv_registered", False):
        return
    for name, obj in {"GSConv": GSConv, "VoVGSCSPC": VoVGSCSPC}.items():
        setattr(tasks, name, obj)
    src = _patch_source(inspect.getsource(tasks.parse_model))
    exec(compile(src, tasks.__file__, "exec"), tasks.__dict__)
    tasks._gv_registered = True
''')

add_code('''%%writefile gv_modules/__init__.py
from .blocks import GSConv, VoVGSCSPC
from .register import register

register()
''')

add_code('''%%writefile models/gv-yolo26-simple.yaml
# GV-YOLO26 Simple: Stock YOLO26 with only backbone changed to GhostConv + VoVGSCSPC
# Neck, P3/P4/P5 detection pyramids, loss, and Detect head remain 100% stock YOLO26.

nc: 2
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# Backbone: Stock YOLO26 with GhostConv & VoVGSCSPC at stages P4 and P5
backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]] # 2
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]] # 4 (Backbone P3)
  - [-1, 1, GhostConv, [512, 3, 2]] # 5-P4/16 (Replaces Conv)
  - [-1, 2, VoVGSCSPC, [512]] # 6 (Replaces C3k2) (Backbone P4)
  - [-1, 1, GhostConv, [1024, 3, 2]] # 7-P5/32 (Replaces Conv)
  - [-1, 2, VoVGSCSPC, [1024]] # 8 (Replaces C3k2) (Backbone P5)
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10 (Backbone P5 output)

# Head: 100% Stock YOLO26 FPN/PAN neck + Detect(P3, P4, P5)
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12 (cat backbone P4 layer 6)
  - [-1, 2, C3k2, [512, True]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15 (cat backbone P3 layer 4)
  - [-1, 2, C3k2, [256, True]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18 (cat head P4 layer 13)
  - [-1, 2, C3k2, [512, True]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21 (cat head P5 layer 10)
  - [-1, 1, C3k2, [1024, True, 0.5, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # 23 Detect(P3, P4, P5)
''')

add_code('''import gv_modules
from ultralytics import YOLO

m = YOLO("models/gv-yolo26-simple.yaml", task="detect")
total_params = sum(p.numel() for p in m.model.parameters())
print(f"GV-YOLO26-Simple initialized OK - Total parameters: {total_params:,}")''')

add_code('''from pathlib import Path
import random
import yaml

print("Scanning /kaggle/input for dataset...")
hits = [p.parent.parent for p in Path("/kaggle/input").rglob("train/images") if (p.parent.parent / "test/images").is_dir()]
assert hits, "No train/images + test/images found anywhere under /kaggle/input"
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
print(open("data/data.yaml").read())''')

add_code('''import gv_modules
from ultralytics import YOLO

print("=== Starting GV-YOLO26-Simple Training ===")
model = YOLO("models/gv-yolo26-simple.yaml", task="detect")

try:
    model.load("yolo26n.pt")
    print("Transferred pretrained weights from yolo26n.pt successfully!")
except Exception as e:
    print("Pretrained transfer notice:", e)

train_results = model.train(
    data="data/data.yaml",
    imgsz=640,
    epochs=100,
    batch=16,
    patience=25,
    optimizer="auto",
    seed=42,
    device=0,
    project="/kaggle/working/runs",
    name="gv_yolo26_simple",
    exist_ok=True,
    plots=True,
    workers=4,
    amp=True,
    verbose=True
)
print("Training completed successfully!")''')

add_code('''import json
import shutil
from pathlib import Path
from ultralytics import YOLO

best_pt = Path("/kaggle/working/runs/gv_yolo26_simple/weights/best.pt")
assert best_pt.exists(), f"best.pt not found at {best_pt}"

print("=== Evaluating Best Model on Test Set ===")
best_model = YOLO(str(best_pt))

val_metrics = best_model.val(
    data="data/data.yaml",
    split="val",
    imgsz=640,
    batch=1,
    conf=0.001,
    iou=0.7,
    device=0,
    plots=False,
    verbose=True
)

test_metrics = best_model.val(
    data="data/data.yaml",
    split="test",
    imgsz=640,
    batch=1,
    conf=0.001,
    iou=0.7,
    device=0,
    plots=True,
    project="/kaggle/working/eval",
    name="test_eval",
    exist_ok=True,
    verbose=True
)

summary = {
    "model": "GV-YOLO26-Simple",
    "params": sum(p.numel() for p in best_model.model.parameters()),
    "val": {
        "precision": float(val_metrics.box.mp),
        "recall": float(val_metrics.box.mr),
        "map50": float(val_metrics.box.map50),
        "map50_95": float(val_metrics.box.map),
    },
    "test": {
        "precision": float(test_metrics.box.mp),
        "recall": float(test_metrics.box.mr),
        "map50": float(test_metrics.box.map50),
        "map50_95": float(test_metrics.box.map),
    }
}

print("\\n" + "=" * 50)
print("FINAL RESULTS SUMMARY")
print("=" * 50)
print(f"Model: {summary['model']} ({summary['params']:,} parameters)")
print(f"Validation mAP@0.5: {summary['val']['map50']:.4f} | mAP@0.5:0.95: {summary['val']['map50_95']:.4f}")
print(f"Test Set   mAP@0.5: {summary['test']['map50']:.4f} | mAP@0.5:0.95: {summary['test']['map50_95']:.4f}")
print("=" * 50)

with open("/kaggle/working/results.json", "w") as f:
    json.dump(summary, f, indent=2)

shutil.copy(str(best_pt), "/kaggle/working/gv_yolo26_simple_best.pt")
print("Saved weights to /kaggle/working/gv_yolo26_simple_best.pt")''')

job_dir = Path("/home/benny/Projects/GV-YOLO26/GV-YOLO26/kaggle_jobs/gv-yolo26-simple-train")
job_dir.mkdir(parents=True, exist_ok=True)
nb_path = job_dir / "gv-yolo26-simple-train.ipynb"
nb_path.write_text(json.dumps(nb, indent=1))
print(f"Generated notebook at: {nb_path} with {len(nb['cells'])} cells")

