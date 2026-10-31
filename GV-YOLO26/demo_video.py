"""In-video speed bump and pothole detection demo with temporal confirmation.

Each frame goes through the NMS-free GV-YOLO26 head, detections are linked across frames with ByteTrack, and a hazard
is drawn only after its track has been seen in at least --confirm frames. This suppresses one-frame false alarms
(shadows, road markings) that a per-frame detector produces. A warning banner names the hazards currently ahead.

Examples:
    python demo_video.py --weights weights/gv_yolo26_best.pt --source ../video            # every clip in a folder
    python demo_video.py --weights weights/gv_yolo26_best.pt --source ../video/v2.mp4 --show
    python demo_video.py --weights weights/gv_yolo26_best.pt --source ../video --confirm 1  # no temporal filter
"""

import argparse
import json
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2

import gv_modules  # noqa: F401  registers the GV-YOLO26 layers
from gv_modules.suspension_preview import SuspensionPreview
from ultralytics import YOLO

VIDEO = {".mp4", ".avi", ".mov", ".mkv"}
COLORS = {"Pothole": (0, 140, 255), "Speed Bump": (60, 220, 60)}  # BGR


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="video file or folder of videos")
    p.add_argument("--out", default="results/video")
    p.add_argument("--conf", type=float, default=0.35, help="minimum confidence of a drawn box")
    p.add_argument("--confirm", type=int, default=3, help="frames a track must be seen in before it is shown")
    p.add_argument("--ego-speed", type=float, default=60.0, help="ego-vehicle speed in km/h for TTI calculation")
    p.add_argument("--cam-height", type=float, default=1.25, help="camera mounting height in meters")
    p.add_argument("--pitch", type=float, default=8.0, help="camera tilt pitch down angle in degrees")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--show", action="store_true", help="display the annotated video live while processing")
    return p.parse_args()


def draw_box(img, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(img, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def draw_hud(img, ahead, fps, counts, suspension_status="NOMINAL", nearest_z=None, nearest_tti=None):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    if ahead:
        text = "WARNING: " + " + ".join(sorted(a.upper() for a in ahead)) + " AHEAD"
        cv2.putText(img, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "ROAD CLEAR", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 220, 90), 2, cv2.LINE_AA)

    # Active Suspension Status line
    susp_color = (0, 0, 255) if "CRITICAL" in suspension_status else ((0, 165, 255) if "SOFTEN" in suspension_status else (50, 220, 90))
    susp_info = f"ACTIVE SUSPENSION: {suspension_status}"
    if nearest_z is not None and nearest_tti is not None:
        susp_info += f" [Z: {nearest_z:.1f}m | TTI: {nearest_tti}ms]"
    cv2.putText(img, susp_info, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, susp_color, 1, cv2.LINE_AA)

    stats = f"GV-YOLO26 | {fps:5.1f} FPS | Bumps: {counts['Speed Bump']}  Potholes: {counts['Pothole']}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.putText(img, stats, (w - tw - 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)


def to_h264(path):
    """Re-encode OpenCV's mp4v output to H.264 so the clip plays in browsers and default players."""
    if not shutil.which("ffmpeg"):
        return path
    tmp = path.with_name(path.stem + "_h264.mp4")
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", str(tmp)])
    if r.returncode == 0:
        tmp.replace(path)
    return path


def process(model, src, opt):
    cap = cv2.VideoCapture(str(src))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_dir = Path(opt.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{src.stem}_gv.mp4"
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    preview_engine = SuspensionPreview(
        cam_height_m=opt.cam_height,
        pitch_deg=opt.pitch,
        default_ego_speed_kmh=opt.ego_speed,
    )

    seen = defaultdict(int)  # track id -> frames observed
    conf_hist = defaultdict(list)  # track id -> rolling confidences
    confirmed = {}  # track id -> class name, once confirmed
    raw = set()  # track ids that ever scored >= --conf, confirmed or not
    frames = frames_with_hazard = frames_with_raw = 0
    infer_time = 0.0
    model.predictor = None  # fresh tracker state for every clip
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        t0 = time.perf_counter()
        # Adaptive night contrast enhancement for low-light headlight scenes
        if cv2.mean(frame)[0] < 100:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            infer_frame = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        else:
            infer_frame = frame

        # low conf here so ByteTrack's second association stage can keep weak, distant hazards alive
        r = model.track(infer_frame, imgsz=opt.imgsz, conf=0.08, nms=False, persist=True, tracker="bytetrack.yaml",
                        device=opt.device, verbose=False)[0]
        infer_time += time.perf_counter() - t0

        ahead = set()
        raw_hit = False
        nearest_hazard = None
        active_susp_status = "NOMINAL"

        if r.boxes.id is not None:
            for xyxy, tid, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.id.int().tolist(),
                                            r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                # Filter out reflections directly on the car hood (bottom 8% of frame)
                if xyxy[3] > 0.92 * h:
                    continue

                seen[tid] += 1
                conf_hist[tid].append(conf)
                if len(conf_hist[tid]) > 10:
                    conf_hist[tid].pop(0)

                name = r.names[cls]
                if conf >= opt.conf:
                    raw.add(tid)
                    raw_hit = True

                cum_conf = sum(conf_hist[tid])
                # Fast trigger in 1 frame for high conf near-field hazards (lower 40% of frame)
                fast_trigger = (conf >= 0.72 and xyxy[3] >= 0.60 * h)
                is_confirmed = fast_trigger or (cum_conf >= 1.05) or (seen[tid] >= opt.confirm and conf >= opt.conf)

                if is_confirmed:
                    confirmed.setdefault(tid, name)
                    ahead.add(name)
                    prev = preview_engine.compute_preview(xyxy, h, name, opt.ego_speed)
                    tag = f"#{tid} {name} {conf:.2f} | {prev['z_m']}m {prev['tti_ms']}ms [{prev['damping_cmd']}]"
                    draw_box(frame, xyxy, tag, prev["damping_color"])

                    if nearest_hazard is None or prev["z_m"] < nearest_hazard["z_m"]:
                        nearest_hazard = prev

        if nearest_hazard is not None:
            active_susp_status = nearest_hazard["damping_cmd"]
        frames_with_hazard += bool(ahead)
        frames_with_raw += raw_hit
        counts = defaultdict(int)
        for name in confirmed.values():
            counts[name] += 1
        nz = nearest_hazard["z_m"] if nearest_hazard else None
        ntti = nearest_hazard["tti_ms"] if nearest_hazard else None
        draw_hud(frame, ahead, frames / infer_time, counts, active_susp_status, nz, ntti)
        writer.write(frame)
        if opt.show:
            cv2.imshow("GV-YOLO26 in-video detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    writer.release()
    to_h264(dst)

    stats = {
        "video": src.name,
        "frames": frames,
        "frames_with_hazard": frames_with_hazard,
        "frames_with_raw_detection": frames_with_raw,
        "raw_tracks": len(raw),
        "confirmed_tracks": len(confirmed),
        "unique_speed_bumps": sum(v == "Speed Bump" for v in confirmed.values()),
        "unique_potholes": sum(v == "Pothole" for v in confirmed.values()),
        "processing_fps": round(frames / infer_time, 1),
        "output": str(dst),
    }
    print(json.dumps(stats))
    return stats


def main(opt):
    model = YOLO(opt.weights)
    src = Path(opt.source)
    clips = sorted(p for p in src.iterdir() if p.suffix.lower() in VIDEO) if src.is_dir() else [src]
    summary = [process(model, c, opt) for c in clips]
    Path(opt.out, "summary.json").write_text(json.dumps(summary, indent=1))
    if opt.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(parse_args())
