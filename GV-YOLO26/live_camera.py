"""Real-time OpenCV camera inference runner for laptop webcams, USB cameras, and phone Wi-Fi streams.

Supports:
- Local laptop integrated webcam: --source 0
- Phone camera streaming over Wi-Fi (IP Webcam / DroidCam / RTSP): --source http://192.168.1.X:8080/video
- Video files or RTSP IP cameras: --source rtsp://...

Controls:
    'q' - Quit
    'e' - Toggle unmarked speed bump CLAHE contrast enhancement
    'm' - Mark / send current detected hazard with GPS to local map server
    's' - Save snapshot to results/snapshots/
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Ensure GV-YOLO26 modules are imported
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gv_modules  # noqa: F401 registers custom layers
from gv_modules.hazard_mapper import estimate_lateral_offset, google_maps_url, project_hazard_gps
from gv_modules.preprocess import enhance_road_surface
from gv_modules.suspension_preview import SuspensionPreview
from ultralytics import YOLO

COLORS = {
    "Pothole": (0, 140, 255),       # BGR: Orange
    "Speed Bump": (60, 220, 60),    # BGR: Bright Green
}


def parse_args():
    p = argparse.ArgumentParser(description="GV-YOLO26 Live OpenCV Camera Runner")
    p.add_argument("--source", default="0", help="Camera index ('0'), phone Wi-Fi URL ('http://192.168.1.X:8080/video'), or video file")
    p.add_argument("--weights", default="weights/gv_yolo26_simple_best.pt", help="Path to trained model weights")
    p.add_argument("--conf", type=float, default=0.20, help="Confidence threshold for candidate detection")
    p.add_argument("--confirm", type=int, default=3, help="Consecutive frames required for confirmed hazard alert")
    p.add_argument("--ego-speed", type=float, default=50.0, help="Vehicle speed in km/h for TTI calculation")
    p.add_argument("--cam-height", type=float, default=1.25, help="Camera height in meters")
    p.add_argument("--pitch", type=float, default=8.0, help="Camera tilt down pitch in degrees")
    p.add_argument("--imgsz", type=int, default=640, help="Model input resolution")
    p.add_argument("--enhance-unmarked", action="store_true", default=True, help="Enable CLAHE surface relief enhancement for unpainted bumps")
    p.add_argument("--map-server", default="http://localhost:8000", help="URL of demo_app map server for geotagging")
    p.add_argument("--lat", type=float, default=12.9716, help="Current or simulated ego latitude")
    p.add_argument("--lon", type=float, default=77.5946, help="Current or simulated ego longitude")
    p.add_argument("--heading", type=float, default=0.0, help="Vehicle heading in degrees (0 = North)")
    return p.parse_args()


def draw_hud(frame, ahead, fps, counts, suspension_status="NOMINAL", nearest_z=None, nearest_tti=None, enhanced=True):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (18, 22, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if ahead:
        text = "WARNING: " + " + ".join(sorted(a.upper() for a in ahead)) + " AHEAD"
        cv2.putText(frame, text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ROAD CLEAR", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 220, 90), 2, cv2.LINE_AA)

    # Active Suspension Status line
    susp_color = (0, 0, 255) if "CRITICAL" in suspension_status else ((0, 165, 255) if "SOFTEN" in suspension_status else (50, 220, 90))
    susp_info = f"ACTIVE SUSPENSION: {suspension_status}"
    if nearest_z is not None and nearest_tti is not None:
        susp_info += f" [Z: {nearest_z:.1f}m | TTI: {nearest_tti}ms]"
    cv2.putText(frame, susp_info, (16, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, susp_color, 1, cv2.LINE_AA)

    mode_tag = " [UNMARKED-ENHANCED]" if enhanced else ""
    stats = f"GV-YOLO26 | {fps:4.1f} FPS | Bumps: {counts['Speed Bump']} Potholes: {counts['Pothole']}{mode_tag}"
    (tw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(frame, stats, (w - tw - 16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)


def main():
    opt = parse_args()
    weights_path = Path(opt.weights)
    if not weights_path.is_absolute() and not weights_path.exists():
        weights_path = ROOT / opt.weights

    if not weights_path.exists():
        print(f"[Error] Weights file not found: {weights_path}")
        sys.exit(1)

    print(f"Loading GV-YOLO26 from {weights_path}...")
    model = YOLO(str(weights_path))

    # Parse camera source
    src = opt.source
    if src.isdigit():
        src = int(src)
        print(f"Opening local camera device index {src}...")
    else:
        print(f"Connecting to video stream: {src}...")

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[Error] Could not open camera source: {opt.source}")
        print("Tip for laptop webcam: ensure /dev/video0 is accessible.")
        print("Tip for phone over Wi-Fi: ensure phone and laptop are on same Wi-Fi and IP address is correct.")
        sys.exit(1)

    preview_engine = SuspensionPreview(
        cam_height_m=opt.cam_height,
        pitch_deg=opt.pitch,
        default_ego_speed_kmh=opt.ego_speed,
    )

    seen = defaultdict(int)
    conf_hist = defaultdict(list)
    confirmed_map = {}
    enhanced_mode = opt.enhance_unmarked

    print("\n--- Camera Stream Active ---")
    print("Press 'q' to exit | 'e' to toggle Unmarked Bump Enhancement | 's' to snapshot\n")

    t_prev = time.perf_counter()
    fps = 0.0

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Video stream ended or frame drop.")
            break

        h, w = raw_frame.shape[:2]

        # 1. Road surface relief & shadow enhancement for unpainted speed bumps
        if enhanced_mode:
            infer_frame = enhance_road_surface(raw_frame)
        else:
            infer_frame = raw_frame

        # 2. GV-YOLO26 NMS-free tracking
        res = model.track(
            source=infer_frame,
            imgsz=opt.imgsz,
            conf=0.10,
            persist=True,
            tracker="bytetrack.yaml",
            nms=False,
            verbose=False,
        )[0]

        ahead_hazards = set()
        counts = {"Pothole": 0, "Speed Bump": 0}
        nearest_hazard = None

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.int().cpu().numpy().tolist()
            confs = res.boxes.conf.cpu().numpy().tolist()
            track_ids = res.boxes.id.int().cpu().numpy().tolist() if res.boxes.id is not None else list(range(len(boxes)))

            for box, tid, cls_id, conf_val in zip(boxes, track_ids, classes, confs):
                if box[3] > 0.92 * h:
                    continue  # hood reflection

                raw_name = res.names.get(cls_id, str(cls_id))
                cls_name = "Speed Bump" if "bump" in raw_name.lower() else ("Pothole" if "pothole" in raw_name.lower() else raw_name)

                seen[tid] += 1
                conf_hist[tid].append(conf_val)
                if len(conf_hist[tid]) > 10:
                    conf_hist[tid].pop(0)

                cum_conf = sum(conf_hist[tid])
                is_confirmed = (seen[tid] >= opt.confirm) or (cum_conf >= 1.05) or (conf_val >= 0.70)

                prev = preview_engine.compute_preview(box, h, cls_name, opt.ego_speed)
                x_lat = estimate_lateral_offset(box, w, prev["z_m"])

                if is_confirmed:
                    confirmed_map[tid] = cls_name
                    ahead_hazards.add(cls_name)
                    counts[cls_name] = counts.get(cls_name, 0) + 1
                    if nearest_hazard is None or prev["z_m"] < nearest_hazard["z_m"]:
                        nearest_hazard = prev

                # Draw bounding box
                should_draw = (conf_val >= opt.conf) or is_confirmed
                if should_draw:
                    color = prev["damping_color"] if is_confirmed else (255, 235, 0)
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(raw_frame, (x1, y1), (x2, y2), color, 3 if is_confirmed else 2)

                    tag = f"#{tid} {cls_name} {conf_val:.2f} | {prev['z_m']:.1f}m [{prev['damping_cmd']}]"
                    cv2.rectangle(raw_frame, (x1, max(0, y1 - 24)), (x1 + len(tag) * 9, y1), color, -1)
                    cv2.putText(raw_frame, tag, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)

        # Compute FPS
        t_now = time.perf_counter()
        dt = t_now - t_prev
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 30.0)

        active_susp = nearest_hazard["damping_cmd"] if nearest_hazard else "NOMINAL"
        nz = nearest_hazard["z_m"] if nearest_hazard else None
        ntti = nearest_hazard["tti_ms"] if nearest_hazard else None

        draw_hud(raw_frame, ahead_hazards, fps, counts, active_susp, nz, ntti, enhanced_mode)

        cv2.imshow("GV-YOLO26 Road Hazard & Suspension Preview", raw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            enhanced_mode = not enhanced_mode
            print(f"[Toggle] Unmarked bump surface enhancement: {'ENABLED' if enhanced_mode else 'DISABLED'}")
        elif key == ord("s"):
            snap_dir = ROOT / "results/snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / f"snap_{int(time.time())}.jpg"
            cv2.imwrite(str(snap_path), raw_frame)
            print(f"[Snapshot] Saved to {snap_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
