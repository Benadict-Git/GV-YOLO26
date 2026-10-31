"""Per-Channel INT8 Quantization-Aware Training (QAT) Preparation Utility for GV-YOLO26.

Configures PyTorch torch.ao.quantization with PerChannelMinMaxObserver for weights
(preserving precision on depthwise and grouped convolutions in GhostConv, GSConv,
and LGDH) and HistogramObserver for activations. Performs dataset calibration and
exports a prepared QAT checkpoint ready for fine-tuning.

Example:
    python tools/prepare_qat.py --weights weights/gv_yolo26_simple_best.pt --data data_local/data.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.ao.quantization as tq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gv_modules  # noqa: E402,F401 registers custom GV-YOLO26 layers
from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Prepare GV-YOLO26 for Per-Channel INT8 QAT")
    p.add_argument(
        "--weights",
        type=str,
        default="weights/gv_yolo26_simple_best.pt",
        help="Path to trained .pt weights",
    )
    p.add_argument(
        "--data",
        type=str,
        default="data_local/data.yaml",
        help="Path to dataset YAML",
    )
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test", "train"],
        help="Dataset split to sample calibration images from",
    )
    p.add_argument(
        "--calibrate-batches",
        type=int,
        default=50,
        help="Number of calibration forward batches",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size during calibration passes",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution (default 640)",
    )
    p.add_argument(
        "--output",
        type=str,
        default="weights/gv_yolo26_qat_prepared.pt",
        help="Output path for prepared QAT model checkpoint",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for QAT preparation and calibration ('cpu' recommended for torch.ao.quantization)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="fbgemm",
        choices=["fbgemm", "qnnpack", "x86"],
        help="Target quantization engine backend",
    )
    return p.parse_args()


def resolve_image_list(data_path, split="val"):
    """Extract list of existing image paths from data YAML or text split files."""
    data_file = Path(data_path)
    if not data_file.exists():
        # Try finding relative to ROOT
        root = Path(__file__).resolve().parents[1]
        alt = root / data_path
        if alt.exists():
            data_file = alt

    if not data_file.exists():
        return []

    try:
        with open(data_file) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to parse YAML {data_file}: {e}")
        return []

    target = cfg.get(split, None)
    if not target:
        target = cfg.get("val", cfg.get("test", cfg.get("train")))

    images = []
    if target:
        target_path = Path(target)
        if not target_path.is_absolute() and "path" in cfg:
            root_path = Path(cfg["path"])
            if not root_path.is_absolute():
                root_path = data_file.parent / root_path
            target_path = root_path / target_path

        if target_path.is_file():
            # Could be a .txt list of image paths
            with open(target_path) as f:
                lines = [line.strip() for line in f if line.strip()]
            for line in lines:
                p = Path(line)
                if not p.is_absolute() and "path" in cfg:
                    p = Path(cfg["path"]) / p
                if p.exists() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    images.append(str(p))
        elif target_path.is_dir():
            for p in sorted(target_path.rglob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    images.append(str(p))

    return images


def preprocess_image(img_path, imgsz=640):
    """Load and letterbox image into normalized tensor (1, 3, imgsz, imgsz)."""
    im = cv2.imread(str(img_path))
    if im is None:
        return None
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dx = (imgsz - nw) // 2
    dy = (imgsz - nh) // 2
    canvas[dy : dy + nh, dx : dx + nw] = im_resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    return tensor


def configure_per_channel_qat(model, backend="fbgemm"):
    """Attach per-channel symmetric weight observers and histogram activation observers."""
    # Obtain base QAT qconfig for specified backend
    try:
        base_qconfig = tq.get_default_qat_qconfig(backend)
    except Exception:
        base_qconfig = tq.get_default_qat_qconfig("fbgemm")

    # Override weight observer to PerChannelMinMaxObserver (essential for GhostConv & LGDH)
    per_channel_weight_observer = tq.FakeQuantize.with_args(
        observer=tq.PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )

    # Use activation observer with reduce_range=False for full 8-bit dynamic range
    activation_observer = tq.FakeQuantize.with_args(
        observer=tq.HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )

    custom_qat_qconfig = base_qconfig._replace(
        weight=per_channel_weight_observer,
        activation=activation_observer,
    )

    model.qconfig = custom_qat_qconfig
    prepared_model = tq.prepare_qat(model, inplace=False)
    return prepared_model


def main(opt):
    weights_path = Path(opt.weights)
    if not weights_path.exists():
        root = Path(__file__).resolve().parents[1]
        alt = root / opt.weights
        if alt.exists():
            weights_path = alt

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {opt.weights}")

    print(f"[*] Loading GV-YOLO26 checkpoint from: {weights_path}")
    yolo_wrapper = YOLO(str(weights_path))
    model = yolo_wrapper.model
    model.to(opt.device)
    model.train()  # prepare_qat requires training mode

    print(f"[*] Configuring Per-Channel QAT with backend: {opt.backend}")
    print("    - Weight observer: PerChannelMinMaxObserver (symmetric INT8 [-128, 127])")
    print("    - Activation observer: HistogramObserver (affine UINT8 [0, 255])")
    print("    - Depthwise & Grouped Conv protection: ENABLED (GhostConv, GSConv, LGDH)")

    prep_model = configure_per_channel_qat(model, backend=opt.backend)

    fake_quants = [m for m in prep_model.modules() if isinstance(m, tq.FakeQuantize)]
    print(f"[*] Total attached FakeQuantize observers: {len(fake_quants)}")

    # Dataset Calibration
    print(f"[*] Searching for calibration images in {opt.data} (split: {opt.split})...")
    image_paths = resolve_image_list(opt.data, split=opt.split)
    if not image_paths:
        print("    [!] No dataset images located; checking fallback data paths...")
        image_paths = resolve_image_list("GV-YOLO26/data_local/data.yaml", split=opt.split)

    if image_paths:
        print(f"    Found {len(image_paths)} images available for calibration.")
        calib_images = image_paths[: opt.calibrate_batches * opt.batch_size]
    else:
        print("    [!] Warning: No physical images found. Using synthetic calibration tensors.")
        calib_images = None

    print(f"[*] Starting calibration over up to {opt.calibrate_batches} batches (batch_size={opt.batch_size})...")
    prep_model.train()
    t0 = time.time()
    batches_run = 0

    with torch.no_grad():
        if calib_images:
            for i in range(0, len(calib_images), opt.batch_size):
                chunk = calib_images[i : i + opt.batch_size]
                tensors = []
                for p in chunk:
                    t = preprocess_image(p, imgsz=opt.imgsz)
                    if t is not None:
                        tensors.append(t)
                if not tensors:
                    continue
                batch = torch.stack(tensors).to(opt.device)
                _ = prep_model(batch)
                batches_run += 1
                if batches_run >= opt.calibrate_batches:
                    break
        else:
            for b in range(opt.calibrate_batches):
                dummy_input = torch.rand(opt.batch_size, 3, opt.imgsz, opt.imgsz, device=opt.device)
                _ = prep_model(dummy_input)
                batches_run += 1

    elapsed = time.time() - t0
    print(f"[*] Calibration completed: {batches_run} batches processed in {elapsed:.2f}s ({batches_run / max(elapsed, 1e-4):.1f} batch/s)")

    # Analyze observer scale statistics
    scales = []
    for fq in fake_quants:
        if hasattr(fq, "scale") and fq.scale is not None and fq.scale.numel() > 0:
            scales.append(fq.scale.detach().cpu())

    if scales:
        per_channel_scales = [s for s in scales if s.numel() > 1]
        print(f"[*] Observer Statistics:")
        print(f"    - Per-channel quantized weight layers: {len(per_channel_scales)}")
        if per_channel_scales:
            sample_pc = per_channel_scales[0]
            print(f"    - Sample layer per-channel scale range: min={sample_pc.min():.5f}, max={sample_pc.max():.5f}, ratio={sample_pc.max() / max(sample_pc.min(), 1e-8):.1f}x")

    # Export prepared QAT checkpoint
    out_path = Path(opt.output)
    if not out_path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        out_path = root / opt.output

    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "state_dict": prep_model.state_dict(),
        "yaml": getattr(model, "yaml", None),
        "names": getattr(yolo_wrapper, "names", {0: "Pothole", 1: "Speed Bump"}),
        "nc": getattr(model, "nc", 2),
        "imgsz": opt.imgsz,
        "backend": opt.backend,
        "qconfig": "PerChannelMinMaxObserver(weight) + HistogramObserver(activation)",
        "calibration_batches": batches_run,
        "source_weights": str(weights_path),
    }

    torch.save(checkpoint, out_path)
    file_size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[✓] Successfully exported prepared QAT model to: {out_path} ({file_size_mb:.2f} MB)")
    print(f"    Downstream training can load 'state_dict' into a prepared QAT model graph and fine-tune for 5-10 epochs.")


if __name__ == "__main__":
    main(parse_args())
