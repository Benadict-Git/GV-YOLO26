"""FastAPI backend for GV-YOLO26 & SBP-YOLO mobile demo with live camera and benchmark streaming."""
import base64
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Register custom architecture modules
ROOT = Path(__file__).resolve().parents[1]
GV_DIR = ROOT / "GV-YOLO26"
DEMO_DIR = ROOT / "demo_app"

if str(GV_DIR) not in sys.path:
    sys.path.insert(0, str(GV_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import gv_modules  # noqa: F401
from gv_modules.preprocess import enhance_road_surface
from gv_modules.hazard_mapper import estimate_lateral_offset, project_hazard_gps

from hazard_store import HazardStore
hazard_store = HazardStore()

from ultralytics import YOLO

app = FastAPI(title="GV-YOLO26 Road Hazard Detection HUD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model paths and registry
GV_WEIGHTS_PATH = GV_DIR / "weights/gv_yolo26_simple_best.pt"
if not GV_WEIGHTS_PATH.exists():
    GV_WEIGHTS_PATH = GV_DIR / "weights_kaggle/benchmark_out/weights/s6_gv_yolo26.pt"

GV_OPT_WEIGHTS_PATH = GV_DIR / "weights_kaggle/benchmark_out/weights/s4_ghost.pt"

# Model definitions
MODELS_REGISTRY = {
    "gv-yolo26": {
        "id": "gv-yolo26",
        "name": "GV-YOLO26",
        "subtitle": "Proposed Backbone-Modified Model",
        "tag": "Proposed (82.8% mAP)",
        "params": "2.08 M (-19.3%)",
        "gflops": "5.3",
        "map50": "82.8%",
        "arch": "YOLO26n + VoVGSCSPC + GhostConv Backbone",
        "path": GV_WEIGHTS_PATH,
        "instance": None,
        "nms_free": True,
        "color_theme": "#00f2fe",
    },
    "gv-yolo26-opt": {
        "id": "gv-yolo26-opt",
        "name": "GV-YOLO26-Opt",
        "subtitle": "High-Capacity Ghost Model",
        "tag": "Optimized (84.1% mAP)",
        "params": "2.08 M",
        "gflops": "5.8",
        "map50": "84.1%",
        "arch": "YOLO26n + C3k2 + GhostConv + P2 + LGDH",
        "path": GV_OPT_WEIGHTS_PATH,
        "instance": None,
        "nms_free": True,
        "color_theme": "#f59e0b",
    },
}

# Load models safely
print("Initializing Detection Models...")
for m_id, m_info in MODELS_REGISTRY.items():
    p = m_info["path"]
    if p and Path(p).exists():
        try:
            print(f"Loading {m_info['name']} from {p}...")
            m_info["instance"] = YOLO(str(p))
            print(f"✅ {m_info['name']} loaded successfully. Classes: {m_info['instance'].names}")
        except Exception as e:
            print(f"⚠️ Warning loading {m_info['name']} ({p}): {e}")
    else:
        print(f"⚠️ Weights file for {m_info['name']} not found at {p}")

# Active model tracker
active_model_id = "gv-yolo26"

# Class colors: Pothole (Amber/Orange), Speed Bump (Neon Green)
COLORS = {
    "Pothole": (0, 140, 255),       # BGR: Orange
    "Speed Bump": (60, 220, 60),    # BGR: Bright Green
}
CANDIDATE_COLORS = {
    "Pothole": (30, 200, 255),     # BGR: Bright Amber
    "Speed Bump": (255, 235, 0),    # BGR: Electric Cyan
}


def normalize_class_name(raw_name: str) -> str:
    """Normalize differences like 'SpeedBump' vs 'Speed Bump'."""
    cleaned = raw_name.replace(" ", "").replace("_", "").lower()
    if "bump" in cleaned:
        return "Speed Bump"
    if "pothole" in cleaned:
        return "Pothole"
    return raw_name


from gv_modules.suspension_preview import SuspensionPreview

preview_engine = SuspensionPreview()

# Session tracker state
tracker_state = {
    "seen": defaultdict(int),       # track_id -> frames seen
    "conf_hist": defaultdict(list), # track_id -> rolling confidence list
    "confirmed": {},                # track_id -> class_name
    "conf_threshold": 0.15,
    "confirm_frames": 1,
    "last_infer_time_ms": 0.0,
    "total_frames": 0,
}


def get_active_model_info():
    """Return active model dict and YOLO instance."""
    info = MODELS_REGISTRY.get(active_model_id) or MODELS_REGISTRY["gv-yolo26"]
    return info, info["instance"]


def draw_hud_box(img, xyxy, label, conf, track_id, confirmed, frames_seen=1, prev_info=None):
    """Draw high-visibility bounding box and HUD tag with candidate distinction and preview telemetry."""
    x1, y1, x2, y2 = map(int, xyxy)
    base_color = COLORS.get(label, (0, 255, 255))
    candidate_color = CANDIDATE_COLORS.get(label, (255, 235, 0))

    if prev_info and confirmed and "damping_color" in prev_info:
        color = prev_info["damping_color"]
    else:
        color = base_color if confirmed else candidate_color
    thickness = 3 if confirmed else 2

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Corner brackets on confirmed hazards
    if confirmed:
        d = min(14, max(4, (x2 - x1) // 5), max(4, (y2 - y1) // 5))
        cv2.line(img, (x1, y1), (x1 + d, y1), (255, 255, 255), 2)
        cv2.line(img, (x1, y1), (x1, y1 + d), (255, 255, 255), 2)
        cv2.line(img, (x2, y2), (x2 - d, y2), (255, 255, 255), 2)
        cv2.line(img, (x2, y2), (x2, y2 - d), (255, 255, 255), 2)

    # Tag text
    status_sym = "●" if confirmed else f"○{frames_seen}"
    if prev_info and confirmed:
        tag = f"#{track_id} {label} {conf:.2f} | {prev_info['z_m']}m {prev_info['tti_ms']}ms [{prev_info['damping_cmd']}] {status_sym}"
    else:
        tag = f"#{track_id} {label} {conf:.2f} {status_sym}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)

    # Tag banner
    tag_y = max(th + 8, y1)
    cv2.rectangle(img, (x1, tag_y - th - 8), (x1 + tw + 10, tag_y), color, -1)
    cv2.putText(img, tag, (x1 + 4, tag_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2, cv2.LINE_AA)


def draw_dashboard_hud(img, ahead_hazards, fps, counts, model_name, suspension_status="NOMINAL", nearest_z=None, nearest_tti=None):
    """Draw automotive HUD top header and warning banner with active suspension telemetry."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (15, 18, 25), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Warning status text
    if ahead_hazards:
        warning_str = "WARNING: " + " + ".join(sorted(h.upper() for h in ahead_hazards)) + " AHEAD"
        cv2.putText(img, warning_str, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img, "ROAD CLEAR", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 220, 90), 2, cv2.LINE_AA)

    # Active Suspension Status line
    susp_color = (0, 0, 255) if "CRITICAL" in suspension_status else ((0, 165, 255) if "SOFTEN" in suspension_status else (50, 220, 90))
    susp_info = f"ACTIVE SUSPENSION: {suspension_status}"
    if nearest_z is not None and nearest_tti is not None:
        susp_info += f" [Z: {nearest_z:.1f}m | TTI: {nearest_tti}ms]"
    cv2.putText(img, susp_info, (16, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, susp_color, 1, cv2.LINE_AA)

    # Telemetry text
    bump_count = counts.get("Speed Bump", 0)
    pothole_count = counts.get("Pothole", 0)
    telemetry = f"{model_name} | {fps:4.1f} FPS | Bumps: {bump_count} Potholes: {pothole_count}"
    (tw, _), _ = cv2.getTextSize(telemetry, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(img, telemetry, (w - tw - 16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)


@app.get("/")
def index():
    """Serve mobile HUD frontend."""
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text())


@app.get("/api/models")
def list_models():
    """List available detection models with architectures and performance metrics."""
    models_list = []
    for m_id, m in MODELS_REGISTRY.items():
        models_list.append({
            "id": m["id"],
            "name": m["name"],
            "subtitle": m["subtitle"],
            "tag": m["tag"],
            "params": m["params"],
            "gflops": m["gflops"],
            "map50": m["map50"],
            "arch": m["arch"],
            "active": (m_id == active_model_id),
            "available": (m["instance"] is not None),
        })
    return JSONResponse({"models": models_list, "active_model": active_model_id})


@app.post("/api/select_model")
def select_model(model_id: str = Form(...)):
    """Switch active model and reset ByteTrack tracker state."""
    global active_model_id
    if model_id not in MODELS_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown model id '{model_id}'")
    if MODELS_REGISTRY[model_id]["instance"] is None:
        raise HTTPException(status_code=400, detail=f"Model '{model_id}' weights not loaded")

    active_model_id = model_id
    reset_tracker()
    return {
        "status": "ok",
        "active_model": active_model_id,
        "name": MODELS_REGISTRY[active_model_id]["name"],
        "params": MODELS_REGISTRY[active_model_id]["params"],
    }


@app.get("/api/clips")
def list_clips():
    """List available benchmark and pre-rendered video clips."""
    clips = [
        {
            "id": "v1",
            "name": "Clip 1: Daytime Rain",
            "scene": "Wet asphalt, rain, painted speed bumps",
            "raw_video": "/video/raw/v1.mp4",
            "annotated_video": "/video/annotated/v1_gv.mp4",
            "duration": "10s",
            "hazards": ["Speed Bump", "Pothole"],
        },
        {
            "id": "v2",
            "name": "Clip 2: Night Scene",
            "scene": "Night headlights, asphalt speed bumps and potholes",
            "raw_video": "/video/raw/v2.mp4",
            "annotated_video": "/video/annotated/v2_gv.mp4",
            "duration": "10s",
            "hazards": ["Speed Bump", "Pothole"],
        },
        {
            "id": "v3",
            "name": "Clip 3: Night Dashcam",
            "scene": "Urban night road hazards",
            "raw_video": "/video/raw/v3.mp4",
            "annotated_video": "/video/annotated/v3_gv.mp4",
            "duration": "10s",
            "hazards": ["Speed Bump", "Pothole"],
        },
        {
            "id": "v4",
            "name": "Clip 4: Dusk Scene",
            "scene": "Low-contrast dusk, potholes and faded bump",
            "raw_video": "/video/raw/v4.mp4",
            "annotated_video": "/video/annotated/v4_gv.mp4",
            "duration": "10s",
            "hazards": ["Speed Bump", "Pothole"],
        },
    ]
    return JSONResponse(clips)


@app.get("/video/raw/{name}")
def get_raw_video(name: str):
    """Stream raw video file with HTTP 206 range request support."""
    path = ROOT / "video" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Raw video not found")
    return FileResponse(path, media_type="video/mp4")


@app.get("/video/annotated/{name}")
def get_annotated_video(name: str):
    """Stream annotated video file with HTTP 206 range request support."""
    path = GV_DIR / "results/video" / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Annotated video not found")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/reset_tracker")
def reset_tracker():
    """Reset ByteTrack tracker states."""
    tracker_state["seen"].clear()
    tracker_state["conf_hist"].clear()
    tracker_state["confirmed"].clear()
    tracker_state["total_frames"] = 0
    for m in MODELS_REGISTRY.values():
        if m["instance"] is not None:
            m["instance"].predictor = None
    return {"status": "ok", "message": "Tracker reset"}


@app.post("/api/predict_frame")
async def predict_frame(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    conf_threshold: float = Form(0.15),
    confirm_frames: int = Form(1),
    ego_speed_kmh: float = Form(60.0),
    model_name: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    heading: Optional[float] = Form(0.0),
    enhance_unmarked: bool = Form(True),
):
    """Run real-time detection, tracking, and temporal confirmation on an uploaded camera frame or image file."""
    global active_model_id
    t0 = time.perf_counter()

    # Model switch if requested
    if model_name and model_name in MODELS_REGISTRY and model_name != active_model_id:
        if MODELS_REGISTRY[model_name]["instance"] is not None:
            active_model_id = model_name
            reset_tracker()

    active_info, model_instance = get_active_model_info()
    if model_instance is None:
        raise HTTPException(status_code=500, detail="No active detection model loaded")

    # Read frame image
    frame = None
    is_static_upload = False

    if file:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        is_static_upload = True
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        decoded = base64.b64decode(image_base64)
        nparr = np.frombuffer(decoded, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image frame provided")

    # For static single-photo uploads, immediate confirmation
    if is_static_upload:
        confirm_frames = 1

    tracker_state["total_frames"] += 1
    tracker_state["conf_threshold"] = conf_threshold
    tracker_state["confirm_frames"] = confirm_frames

    # Road surface relief and contrast enhancement for unmarked speed bumps
    if enhance_unmarked:
        infer_frame = enhance_road_surface(frame)
    else:
        is_night = cv2.mean(frame)[0] < 100
        if is_night:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            infer_frame = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        else:
            infer_frame = frame

    # Determine NMS setting based on model architecture
    track_kwargs = {
        "source": infer_frame,
        "imgsz": 640,
        "conf": 0.08 if not is_static_upload else 0.12,
        "persist": not is_static_upload,
        "tracker": "bytetrack.yaml",
        "verbose": False,
    }
    if active_info["nms_free"]:
        track_kwargs["nms"] = False

    res = model_instance.track(**track_kwargs)[0]

    ahead_hazards = set()
    counts = {"Pothole": 0, "Speed Bump": 0}
    detections = []
    nearest_hazard = None

    seen = tracker_state["seen"]
    conf_hist = tracker_state["conf_hist"]
    confirmed_map = tracker_state["confirmed"]

    if res.boxes is not None and len(res.boxes) > 0:
        boxes_xyxy = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.int().cpu().numpy().tolist()
        confs = res.boxes.conf.cpu().numpy().tolist()

        if res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().numpy().tolist()
        else:
            # Fallback track IDs for static single images or untracked frames
            track_ids = [i + 1 for i in range(len(boxes_xyxy))]

        for box, tid, cls_id, conf_val in zip(boxes_xyxy, track_ids, classes, confs):
            # Filter reflections directly on the car hood/dash (bottom 8% of frame)
            if box[3] > 0.92 * frame.shape[0]:
                continue

            raw_name = res.names.get(cls_id, str(cls_id))
            cls_name = normalize_class_name(raw_name)

            seen[tid] += 1
            conf_hist[tid].append(conf_val)
            if len(conf_hist[tid]) > 10:
                conf_hist[tid].pop(0)

            cum_conf = sum(conf_hist[tid])
            # Fast trigger for high confidence near-field hazards (lower 40% of frame)
            fast_trigger = (conf_val >= 0.72 and box[3] >= 0.60 * frame.shape[0])
            is_confirmed = fast_trigger or (cum_conf >= 1.05) or ((seen[tid] >= confirm_frames) and (conf_val >= conf_threshold))

            # Auto-confirm on static image uploads if confidence meets threshold
            if is_static_upload and conf_val >= conf_threshold:
                is_confirmed = True

            prev = preview_engine.compute_preview(box, frame.shape[0], cls_name, ego_speed_kmh)
            x_lat_m = estimate_lateral_offset(box, frame.shape[1], prev["z_m"])

            geotag_info = None
            if is_confirmed and lat is not None and lon is not None:
                try:
                    recorded_h = hazard_store.record_hazard(
                        class_name=cls_name,
                        ego_lat=float(lat),
                        ego_lon=float(lon),
                        heading_deg=float(heading or 0.0),
                        dist_z_m=float(prev["z_m"]),
                        dist_x_m=float(x_lat_m),
                        confidence=float(conf_val),
                        ego_speed_kmh=float(ego_speed_kmh),
                    )
                    geotag_info = {
                        "hazard_id": recorded_h["id"],
                        "lat": recorded_h["lat"],
                        "lon": recorded_h["lon"],
                        "status": recorded_h["status"],
                        "sightings": recorded_h["sightings"],
                        "google_maps_url": recorded_h["google_maps_url"],
                    }
                except Exception as e:
                    print(f"[Geotag Warning] Failed to record hazard: {e}")

            if is_confirmed:
                confirmed_map[tid] = cls_name
                ahead_hazards.add(cls_name)
                counts[cls_name] = counts.get(cls_name, 0) + 1
                if nearest_hazard is None or prev["z_m"] < nearest_hazard["z_m"]:
                    nearest_hazard = prev

            # Draw box on frame: draw if confirmed OR if confidence meets threshold
            should_draw = (conf_val >= conf_threshold) or is_confirmed
            if should_draw:
                draw_hud_box(frame, box, cls_name, conf_val, tid, is_confirmed, seen[tid], prev_info=prev)

            detections.append({
                "track_id": tid,
                "class_name": cls_name,
                "confidence": round(float(conf_val), 3),
                "confirmed": bool(is_confirmed),
                "frames_observed": seen[tid],
                "box": [round(float(c), 1) for c in box],
                "z_m": prev["z_m"],
                "tti_ms": prev["tti_ms"],
                "damping_cmd": prev["damping_cmd"],
                "geotag": geotag_info,
            })

    # Measure latency
    t_end = time.perf_counter()
    latency_ms = (t_end - t0) * 1000.0
    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
    tracker_state["last_infer_time_ms"] = latency_ms

    active_susp_status = nearest_hazard["damping_cmd"] if nearest_hazard else "NOMINAL"
    nz = nearest_hazard["z_m"] if nearest_hazard else None
    ntti = nearest_hazard["tti_ms"] if nearest_hazard else None

    # Overlay HUD dashboard
    draw_dashboard_hud(frame, ahead_hazards, fps, counts, active_info["name"], active_susp_status, nz, ntti)

    # Encode annotated frame to JPEG base64
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    encoded_frame = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "annotated_frame": f"data:image/jpeg;base64,{encoded_frame}",
        "hazards_ahead": sorted(list(ahead_hazards)),
        "alert": len(ahead_hazards) > 0,
        "counts": counts,
        "detections": detections,
        "suspension": {
            "damping_cmd": active_susp_status,
            "nearest_z_m": nz,
            "nearest_tti_ms": ntti,
            "ego_speed_kmh": round(ego_speed_kmh, 1),
        },
        "fps": round(fps, 1),
        "latency_ms": round(latency_ms, 1),
        "frame_number": tracker_state["total_frames"],
        "active_model": {
            "id": active_info["id"],
            "name": active_info["name"],
            "params": active_info["params"],
            "gflops": active_info["gflops"],
            "map50": active_info["map50"],
            "arch": active_info["arch"],
        },
    })


@app.get("/api/status")
def get_status():
    """System and model status."""
    active_info, _ = get_active_model_info()
    return {
        "status": "online",
        "active_model": active_info["name"],
        "active_model_id": active_model_id,
        "parameters": active_info["params"],
        "gflops": active_info["gflops"],
        "map50": active_info["map50"],
        "architecture": active_info["arch"],
        "available_models": list(MODELS_REGISTRY.keys()),
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/hazards")
def list_hazards(class_name: Optional[str] = None):
    """List all crowdsourced hazards with spatial coordinates and Google Maps links."""
    return JSONResponse(hazard_store.get_all(class_name))


@app.get("/api/hazards/geojson")
def get_hazards_geojson():
    """Return all hazards formatted as RFC 7946 standard GeoJSON FeatureCollection."""
    return JSONResponse(hazard_store.to_geojson())


@app.get("/api/hazards/kml")
def download_hazards_kml():
    """Download hazards as KML file for direct import into Google My Maps or Google Earth."""
    kml_str = hazard_store.to_kml()
    return Response(
        content=kml_str,
        media_type="application/vnd.google-earth.kml+xml",
        headers={"Content-Disposition": "attachment; filename=gv_yolo26_hazards.kml"},
    )


@app.post("/api/hazards/report")
async def report_hazard(
    class_name: str = Form(...),
    lat: float = Form(...),
    lon: float = Form(...),
    heading: float = Form(0.0),
    z_m: float = Form(15.0),
    x_m: float = Form(0.0),
    confidence: float = Form(0.85),
    ego_speed_kmh: float = Form(60.0),
):
    """Manually or externally report a hazard with GPS coordinates for mapping."""
    res = hazard_store.record_hazard(
        class_name=normalize_class_name(class_name),
        ego_lat=lat,
        ego_lon=lon,
        heading_deg=heading,
        dist_z_m=z_m,
        dist_x_m=x_m,
        confidence=confidence,
        ego_speed_kmh=ego_speed_kmh,
    )
    return {"status": "ok", "hazard": res}


@app.delete("/api/hazards/clear")
def clear_hazards():
    """Clear all recorded hazards from the spatial store."""
    hazard_store.clear()
    return {"status": "ok", "message": "All hazard records cleared"}


@app.get("/api/hazards/summary")
def get_hazards_summary():
    """Return summary statistics of recorded speed bumps and potholes."""
    hazards = hazard_store.get_all()
    speed_bumps = [h for h in hazards if h["class_name"] == "Speed Bump"]
    potholes = [h for h in hazards if h["class_name"] == "Pothole"]
    verified = [h for h in hazards if h["status"] == "VERIFIED"]
    return {
        "total": len(hazards),
        "speed_bumps": len(speed_bumps),
        "potholes": len(potholes),
        "verified": len(verified),
        "candidates": len(hazards) - len(verified),
    }


# Mount static assets
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
