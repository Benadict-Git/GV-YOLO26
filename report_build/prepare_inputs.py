"""Prepare all JSON and image inputs for report building."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "report_build"
INPUTS_DIR = BUILD_DIR / "inputs"
INPUTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. benchmark.json
b = json.loads((ROOT / "GV-YOLO26/weights_kaggle/benchmark_out/benchmark.json").read_text())
kd_scratch = json.loads((ROOT / "GV-YOLO26/data_local/kd_scratch_acc.json").read_text())
kd_ft = json.loads((ROOT / "GV-YOLO26/data_local/kd_ft_acc.json").read_text())
b["gv_yolo26_distilled"] = kd_scratch
b["gv_yolo26_kd_ft"] = kd_ft
s1_pt = ROOT / "GV-YOLO26/weights_kaggle/benchmark_out/weights/s1_yolo26n.pt"
s6_pt = ROOT / "GV-YOLO26/weights_kaggle/benchmark_out/weights/s6_gv_yolo26.pt"
b["pt_mb"] = {
    "s1_yolo26n": s1_pt.stat().st_size / (1024 * 1024),
    "s6_gv_yolo26": s6_pt.stat().st_size / (1024 * 1024),
}
(INPUTS_DIR / "benchmark.json").write_text(json.dumps(b, indent=2))

# 2. dataset_stats.json
ds = json.loads((ROOT / "GV-YOLO26/data_local/dataset_stats.json").read_text())
(INPUTS_DIR / "dataset_stats.json").write_text(json.dumps(ds, indent=2))

# 3. video_demo.json
v_summary = json.loads((ROOT / "GV-YOLO26/results/video/summary.json").read_text())
SCENES = {
    "v1.mp4": "daytime, rain",
    "v2.mp4": "night",
    "v3.mp4": "night",
    "v4.mp4": "dusk",
}
for c in v_summary:
    c["scene"] = SCENES.get(c["video"], "road")
vid = {"clips": v_summary}
(INPUTS_DIR / "video_demo.json").write_text(json.dumps(vid, indent=2))

# 4. env.json
env = {
    "python": "3.13.15",
    "torch": "2.11.0+cu128",
    "torch_short": "2.11",
    "cuda": "12.8",
    "tensorrt": "11.4",
    "albumentations": "2.0.8",
    "cv2": "4.14.0",
    "numpy": "2.1.3",
    "ultralytics": "8.4.156",
}
(INPUTS_DIR / "env.json").write_text(json.dumps(env, indent=2))

# 5. Figures
for f in ["fig4_8.png", "fig4_9.png", "fig4_10.png"]:
    src = ROOT / "GV-YOLO26/results/figures" / f
    dst = INPUTS_DIR / f
    shutil.copy2(src, dst)

# 6. narrative.json
sys.path.insert(0, str(BUILD_DIR))
import narrative

kd_info = {
    "scratch_best_epoch": 2,
    "scratch_epochs": 32,
    "ft_best_epoch": 1,
    "ft_epochs": 31,
    "ft_min_kd": -1260,
    "invalid_pct": 0.9,
    "giou_min": -10,
    "giou_max": 1,
    "v4_bump_max": 0.36,
}
box_stats = {
    "bump_width_pct": 72.0,
    "bump_small_pct": 0.2,
    "pothole_small_pct": 46.5,
}

N = narrative.build(b, ds, vid, "s6_gv_yolo26", "stage 6 (GV-YOLO26)", kd_info, box_stats)
(INPUTS_DIR / "narrative.json").write_text(json.dumps(N, indent=2))

print("Prepared all inputs successfully!")
