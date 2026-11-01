"""Synthetic Map Simulation Demo for GV-YOLO26 Hazard Detection & Active Suspension Preview.

Simulates an ego-vehicle driving along a route, detecting painted and unmarked speed bumps
and potholes using monocular pinhole distance geometry and geodesic GPS projection.
Generates a standalone, fully-interactive HTML map with vehicle animation, preview radar beam,
CAN suspension telemetry, and one-click Google Maps links.
"""

import argparse
import json
import math
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Path setup
ROOT = Path(__file__).resolve().parents[1]
GV_DIR = ROOT / "GV-YOLO26"
DEMO_DIR = ROOT / "demo_app"

if str(GV_DIR) not in sys.path:
    sys.path.insert(0, str(GV_DIR))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import gv_modules  # noqa: F401
from gv_modules.hazard_mapper import (
    google_maps_dir_url,
    google_maps_url,
    haversine_distance_m,
    project_hazard_gps,
)
from gv_modules.suspension_preview import SuspensionPreview
from hazard_store import HazardStore


# Pre-defined realistic ground truth hazards along a simulation corridor (Bangalore, MG Road arterial)
SIMULATED_CORRIDOR = {
    "center": [12.972500, 77.595000],
    "waypoints": [
        [12.971000, 77.594200],  # Start: South
        [12.972200, 77.594700],  # Leg 1
        [12.973500, 77.595200],  # Leg 2
        [12.974800, 77.595700],  # Leg 3
        [12.976000, 77.596200],  # End: North
    ],
    "ground_truth_hazards": [
        {
            "id": "gt_bump_1",
            "class_name": "Speed Bump",
            "type": "painted",
            "lat": 12.971850,
            "lon": 77.594550,
            "description": "Standard Yellow/White Striped Speed Table",
        },
        {
            "id": "gt_hole_1",
            "class_name": "Pothole",
            "type": "depression",
            "lat": 12.972900,
            "lon": 77.594980,
            "description": "Deep Asphalt Depression (12cm depth)",
        },
        {
            "id": "gt_bump_2",
            "class_name": "Speed Bump",
            "type": "unmarked",
            "lat": 12.974100,
            "lon": 77.595450,
            "description": "Unmarked Black Asphalt Speed Hump (Low Contrast)",
        },
        {
            "id": "gt_bump_3",
            "class_name": "Speed Bump",
            "type": "unmarked",
            "lat": 12.975400,
            "lon": 77.595950,
            "description": "Unpainted Concrete Rumble Speed Breaker",
        },
    ],
}


def interpolate_route(waypoints: List[List[float]], step_distance_m: float = 2.0) -> List[Dict[str, Any]]:
    """Interpolate waypoints into smooth trajectory steps with heading and speed."""
    route_points = []
    r_earth = 6371000.0

    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]

        dist_m = haversine_distance_m(lat1, lon1, lat2, lon2)
        steps = max(2, int(dist_m / step_distance_m))

        d_lat = lat2 - lat1
        d_lon = lon2 - lon1
        heading = (math.degrees(math.atan2(d_lon, d_lat)) + 360.0) % 360.0

        for s in range(steps):
            frac = s / steps
            curr_lat = lat1 + d_lat * frac
            curr_lon = lon1 + d_lon * frac
            speed_kmh = 45.0 + 10.0 * math.sin(s * 0.15)  # Realistic 40-55 km/h driving speed

            route_points.append({
                "step": len(route_points),
                "lat": round(curr_lat, 6),
                "lon": round(curr_lon, 6),
                "heading": round(heading, 1),
                "speed_kmh": round(speed_kmh, 1),
            })

    return route_points


def run_simulation(route_points: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], HazardStore]:
    """Execute forward preview simulation: detects hazards, calculates TTI, and clusters in HazardStore."""
    preview_engine = SuspensionPreview(cam_height_m=1.25, pitch_deg=8.0)
    store = HazardStore(data_path=DEMO_DIR / "data" / "simulated_hazards.json", cluster_radius_m=6.0)
    store.clear()

    gt_hazards = SIMULATED_CORRIDOR["ground_truth_hazards"]
    timeline = []

    print("\n--- Running Synthetic Hazard Detection & Mapping Simulation ---")
    print(f"Total Route Steps: {len(route_points)} (~{len(route_points) * 2} meters)\n")

    for pt in route_points:
        ego_lat, ego_lon = pt["lat"], pt["lon"]
        heading = pt["heading"]
        speed_kmh = pt["speed_kmh"]

        visible_detections = []
        highest_alert_cmd = "NOMINAL"
        nearest_z = 999.0
        nearest_tti = 9999

        for gt in gt_hazards:
            dist_to_gt = haversine_distance_m(ego_lat, ego_lon, gt["lat"], gt["lon"])

            # Camera line of sight detection zone (between 3m and 35m ahead)
            if 3.0 <= dist_to_gt <= 35.0:
                # Calculate relative bearing to ensure hazard is in forward camera cone (+-25 deg)
                d_lat = gt["lat"] - ego_lat
                d_lon = gt["lon"] - ego_lon
                angle_to_gt = (math.degrees(math.atan2(d_lon, d_lat)) + 360.0) % 360.0
                diff_angle = abs((angle_to_gt - heading + 180.0) % 360.0 - 180.0)

                if diff_angle <= 25.0:
                    z_m = dist_to_gt * math.cos(math.radians(diff_angle))
                    x_m = dist_to_gt * math.sin(math.radians(diff_angle))

                    # Confidence reflects painted vs unpainted detection
                    conf = 0.88 if gt["type"] == "painted" else (0.91 if gt["class_name"] == "Pothole" else 0.76)

                    # Compute Suspension Preview & Damper Command
                    # Mock box for preview calculation
                    dummy_box = (200.0, 300.0, 440.0, 420.0)
                    prev = preview_engine.compute_preview(dummy_box, 640, gt["class_name"], speed_kmh)
                    # Override z_m with simulated exact geometry
                    prev["z_m"] = round(z_m, 1)
                    speed_ms = max(1.0, speed_kmh / 3.6)
                    prev["tti_ms"] = int((z_m / speed_ms) * 1000.0)

                    if z_m < 15.0:
                        prev["damping_cmd"] = "PRELOAD_CRITICAL"
                    elif 15.0 <= z_m <= 30.0:
                        prev["damping_cmd"] = "SOFTEN_FAST"
                    else:
                        prev["damping_cmd"] = "PREVIEW_TRACK"

                    # Record in crowdsourced spatial HazardStore with 6m clustering
                    rec = store.record_hazard(
                        class_name=gt["class_name"],
                        ego_lat=ego_lat,
                        ego_lon=ego_lon,
                        heading_deg=heading,
                        dist_z_m=z_m,
                        dist_x_m=x_m,
                        confidence=conf,
                        ego_speed_kmh=speed_kmh,
                    )

                    visible_detections.append({
                        "hazard_id": rec["id"],
                        "class_name": gt["class_name"],
                        "subtype": gt["type"],
                        "description": gt["description"],
                        "z_m": prev["z_m"],
                        "tti_ms": prev["tti_ms"],
                        "damping_cmd": prev["damping_cmd"],
                        "status": rec["status"],
                        "sightings": rec["sightings"],
                        "lat": rec["lat"],
                        "lon": rec["lon"],
                        "google_maps_url": rec["google_maps_url"],
                    })

                    if z_m < nearest_z:
                        nearest_z = z_m
                        nearest_tti = prev["tti_ms"]
                        highest_alert_cmd = prev["damping_cmd"]

        # Log timeline frame
        frame_state = {
            "step": pt["step"],
            "ego_lat": ego_lat,
            "ego_lon": ego_lon,
            "heading": heading,
            "speed_kmh": speed_kmh,
            "detections": visible_detections,
            "suspension_cmd": highest_alert_cmd,
            "nearest_z_m": nearest_z if nearest_z < 900.0 else None,
            "nearest_tti_ms": nearest_tti if nearest_tti < 9000 else None,
        }
        timeline.append(frame_state)

        if visible_detections:
            d = visible_detections[0]
            print(f"[Step {pt['step']:3d}] Ego: ({ego_lat:.5f}, {ego_lon:.5f}) | {speed_kmh:4.1f} km/h | "
                  f"{d['subtype'].upper()} {d['class_name']} @ {d['z_m']:4.1f}m (TTI: {d['tti_ms']:4d}ms) -> [{d['damping_cmd']}] "
                  f"Sightings: {d['sightings']} ({d['status']})")

    summary = store.to_geojson()["metadata"]
    print("\n--- Simulation Summary ---")
    print(f"Total Unique Hazards Geotagged : {summary['total_hazards']}")
    print(f"Verified Hazards (Sightings >= 2): {summary['verified_hazards']}")
    print(f"Speed Bumps Mapped             : {summary['speed_bumps']}")
    print(f"Potholes Mapped                : {summary['potholes']}")

    return timeline, store


def generate_interactive_map_html(timeline: List[Dict[str, Any]], store: HazardStore, out_file: Path) -> Path:
    """Generate a rich, standalone HTML interactive simulation map using Leaflet and Google Maps links."""
    geojson_data = store.to_geojson()
    kml_data = store.to_kml()

    # Save standalone KML and GeoJSON
    out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    kml_file = out_dir / "synthetic_hazards.kml"
    kml_file.write_text(kml_data, encoding="utf-8")

    geojson_file = out_dir / "synthetic_hazards.geojson"
    geojson_file.write_text(json.dumps(geojson_data, indent=2), encoding="utf-8")

    center = SIMULATED_CORRIDOR["center"]
    waypoints = SIMULATED_CORRIDOR["waypoints"]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>GV-YOLO26 Synthetic Hazard Map Simulation</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --brand: #00f2fe;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      background: #182234;
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      z-index: 100;
    }}
    .header-left {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .badge {{
      font-size: 11px;
      font-weight: 700;
      padding: 3px 8px;
      border-radius: 4px;
      background: rgba(0, 242, 254, 0.15);
      color: var(--brand);
      border: 1px solid rgba(0, 242, 254, 0.3);
      font-family: 'JetBrains Mono', monospace;
    }}
    .header-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .btn {{
      background: var(--card-bg);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }}
    .btn:hover {{ background: #283548; border-color: var(--brand); }}
    .btn-primary {{
      background: #2563eb;
      border-color: #3b82f6;
      color: #ffffff;
    }}
    .btn-primary:hover {{ background: #1d4ed8; }}
    .main-layout {{
      display: flex;
      flex: 1;
      position: relative;
      overflow: hidden;
    }}
    #map {{
      flex: 1;
      height: 100%;
      background: #0b1120;
    }}
    /* Floating HUD telemetry card */
    .telemetry-panel {{
      position: absolute;
      top: 16px;
      right: 16px;
      width: 320px;
      background: rgba(15, 23, 42, 0.88);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 16px;
      z-index: 1000;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }}
    .panel-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }}
    .hud-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .hud-box {{
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 10px;
    }}
    .hud-lbl {{ font-size: 10px; color: var(--muted); }}
    .hud-val {{ font-size: 16px; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }}
    .susp-alert-box {{
      border-radius: 6px;
      padding: 10px;
      text-align: center;
      font-weight: 700;
      font-size: 13px;
      font-family: 'JetBrains Mono', monospace;
      margin-bottom: 12px;
      border: 1px solid;
    }}
    .susp-NOMINAL {{ background: rgba(16, 185, 129, 0.15); border-color: var(--green); color: var(--green); }}
    .susp-SOFTEN_FAST {{ background: rgba(245, 158, 11, 0.2); border-color: var(--amber); color: var(--amber); }}
    .susp-PRELOAD_CRITICAL {{ background: rgba(239, 68, 68, 0.25); border-color: var(--red); color: var(--red); animation: pulse 1s infinite; }}
    @keyframes pulse {{
      0% {{ opacity: 0.8; }} 50% {{ opacity: 1; }} 100% {{ opacity: 0.8; }}
    }}
    /* Playback controls bottom bar */
    .controls-bar {{
      position: absolute;
      bottom: 20px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(15, 23, 42, 0.92);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(255, 255, 255, 0.15);
      border-radius: 30px;
      padding: 8px 20px;
      display: flex;
      align-items: center;
      gap: 14px;
      z-index: 1000;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    }}
    .ctrl-btn {{
      background: none;
      border: none;
      color: var(--text);
      cursor: pointer;
      font-size: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 6px;
      border-radius: 50%;
      transition: background 0.15s;
    }}
    .ctrl-btn:hover {{ background: rgba(255, 255, 255, 0.1); }}
    .slider-wrap {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
    }}
    input[type=range] {{
      accent-color: var(--brand);
      cursor: pointer;
      width: 140px;
    }}
    /* Custom Map Markers */
    .hazard-pin {{
      display: flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
      cursor: pointer;
    }}
    .vehicle-marker {{
      transition: transform 0.1s linear;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-left">
      <h2 style="font-size: 15px; font-weight: 700;">GV-YOLO26 Synthetic Hazard Map Simulation</h2>
      <span class="badge">NMS-FREE PREVIEW</span>
      <span class="badge" style="color: var(--green); border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.15);">GPS GEODESIC</span>
    </div>
    <div class="header-actions">
      <a href="synthetic_hazards.kml" download="synthetic_hazards.kml" class="btn btn-primary">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Export Google My Maps (KML)
      </a>
      <a href="synthetic_hazards.geojson" download="synthetic_hazards.geojson" class="btn">
        Download GeoJSON
      </a>
    </div>
  </header>

  <div class="main-layout">
    <div id="map"></div>

    <!-- Live Telemetry Panel -->
    <div class="telemetry-panel">
      <div class="panel-title">
        <span>Active Suspension Preview</span>
        <span id="hudStep">Step 0/0</span>
      </div>

      <div class="susp-alert-box susp-NOMINAL" id="hudSuspensionBox">
        ACTIVE SUSPENSION: NOMINAL
      </div>

      <div class="hud-grid">
        <div class="hud-box">
          <div class="hud-lbl">VEHICLE SPEED</div>
          <div class="hud-val" id="hudSpeed" style="color: var(--brand);">0 km/h</div>
        </div>
        <div class="hud-box">
          <div class="hud-lbl">PREVIEW DISTANCE</div>
          <div class="hud-val" id="hudDistance">-- m</div>
        </div>
        <div class="hud-box">
          <div class="hud-lbl">TIME TO IMPACT</div>
          <div class="hud-val" id="hudTti">-- ms</div>
        </div>
        <div class="hud-box">
          <div class="hud-lbl">MAPPED HAZARDS</div>
          <div class="hud-val" id="hudHazardCount" style="color: var(--green);">0</div>
        </div>
      </div>

      <div style="font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 8px;">
        <div style="margin-bottom: 4px;"><b>Camera:</b> 1080p @ 30 FPS (HFOV 65°)</div>
        <div style="margin-bottom: 4px;"><b>Model:</b> GV-YOLO26-Simple (2.08M, 5.8 GFLOPs)</div>
        <div><b>Spatial Deduplication:</b> 6.0m DBSCAN Radius</div>
      </div>
    </div>

    <!-- Playback Controls Bar -->
    <div class="controls-bar">
      <button class="ctrl-btn" id="btnPlayPause" onclick="togglePlayPause()" title="Play / Pause">
        <svg id="playIcon" width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
      </button>
      <button class="ctrl-btn" onclick="stepBackward()" title="Step Backward">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"></polyline></svg>
      </button>
      <button class="ctrl-btn" onclick="stepForward()" title="Step Forward">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
      </button>
      <div class="slider-wrap">
        <input type="range" id="timelineSlider" min="0" max="{len(timeline) - 1}" value="0" oninput="seekTo(this.value)">
      </div>
      <button class="btn" onclick="restartSimulation()" style="padding: 4px 10px; font-size: 11px;">Reset</button>
      <button class="btn" id="btnSpeed" onclick="cycleSpeed()" style="padding: 4px 10px; font-size: 11px;">1.0x</button>
    </div>
  </div>

  <script>
    // Embedded simulation payload
    const timeline = {json.dumps(timeline)};
    const waypoints = {json.dumps(waypoints)};
    const center = {json.dumps(center)};

    // Leaflet map setup
    const map = L.map('map', {{
      center: center,
      zoom: 16,
      zoomControl: true
    }});

    // Modern Dark CartoDB / OSM tiles
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap &copy; CARTO | GV-YOLO26 Simulation'
    }}).addTo(map);

    // Draw planned driving corridor route polyline
    const routeLine = L.polyline(waypoints, {{
      color: '#3b82f6',
      weight: 5,
      opacity: 0.6,
      dashArray: '8, 8'
    }}).addTo(map);

    // Vehicle marker icon (Car with heading arrow)
    const carIcon = L.divIcon({{
      className: 'vehicle-marker',
      html: `
        <div style="position: relative; width: 28px; height: 28px; display: flex; align-items: center; justify-content: center;">
          <div style="position: absolute; width: 28px; height: 28px; background: rgba(0, 242, 254, 0.25); border-radius: 50%; animation: pulse 1.5s infinite;"></div>
          <div id="carArrow" style="width: 20px; height: 20px; background: #0284c7; border: 2.5px solid #ffffff; border-radius: 50%; box-shadow: 0 0 10px rgba(0,242,254,0.8); display: flex; align-items: center; justify-content: center; transform: rotate(0deg);">
            <div style="width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent; border-bottom: 7px solid #ffffff; transform: translateY(-1px);"></div>
          </div>
        </div>
      `,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    }});

    const vehicleMarker = L.marker(waypoints[0], {{ icon: carIcon }}).addTo(map);

    // Radar Preview Cone
    const radarPolygon = L.polygon([], {{
      color: '#00f2fe',
      fillColor: '#00f2fe',
      fillOpacity: 0.12,
      weight: 1.5,
      dashArray: '4, 4'
    }}).addTo(map);

    // Hazard Pins Layer
    const hazardMarkers = {{}};

    // Playback state
    let currentIndex = 0;
    let isPlaying = false;
    let playSpeed = 1.0;
    let timer = null;

    function updateSimulationFrame(idx) {{
      currentIndex = idx;
      const frame = timeline[idx];
      if (!frame) return;

      // 1. Move vehicle
      const lat = frame.ego_lat;
      const lon = frame.ego_lon;
      const heading = frame.heading;
      vehicleMarker.setLatLng([lat, lon]);

      // Rotate heading arrow
      const arrowEl = document.getElementById('carArrow');
      if (arrowEl) {{
        arrowEl.style.transform = `rotate(${{heading}}deg)`;
      }}

      // 2. Draw 35m radar preview cone ahead
      const fovRad = (65.0 * Math.PI) / 180.0;
      const rangeM = 35.0;
      const rEarth = 6371000.0;
      const hRad = (heading * Math.PI) / 180.0;

      const leftRad = hRad - fovRad / 2;
      const rightRad = hRad + fovRad / 2;

      const leftLat = lat + ((rangeM * Math.cos(leftRad)) / rEarth) * (180 / Math.PI);
      const leftLon = lon + ((rangeM * Math.sin(leftRad)) / (rEarth * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI);

      const rightLat = lat + ((rangeM * Math.cos(rightRad)) / rEarth) * (180 / Math.PI);
      const rightLon = lon + ((rangeM * Math.sin(rightRad)) / (rEarth * Math.cos((lat * Math.PI) / 180))) * (180 / Math.PI);

      radarPolygon.setLatLngs([[lat, lon], [leftLat, leftLon], [rightLat, rightLon]]);

      // 3. Render or update detected hazards on map
      frame.detections.forEach(det => {{
        const hid = det.hazard_id;
        const isBump = det.class_name === 'Speed Bump';
        const isUnmarked = det.subtype === 'unmarked';
        const isHole = det.class_name === 'Pothole';

        let color = '#10b981'; // Green (Painted Speed Bump)
        let label = '⚡ SPEED BUMP';
        if (isUnmarked) {{
          color = '#f59e0b'; // Amber (Unmarked Asphalt Bump)
          label = '⚡ UNMARKED BUMP';
        }} else if (isHole) {{
          color = '#ef4444'; // Red (Pothole)
          label = '⚠️ POTHOLE';
        }}

        if (!hazardMarkers[hid]) {{
          const icon = L.divIcon({{
            className: 'hazard-pin',
            html: `
              <div style="background: ${{color}}; color: #000000; font-weight: 700; font-size: 10px; font-family: 'JetBrains Mono', monospace; padding: 3px 8px; border-radius: 12px; border: 2px solid #ffffff; box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
                ${{label}} [Z:${{det.z_m}}m]
              </div>
            `,
            iconAnchor: [45, 12]
          }});

          const marker = L.marker([det.lat, det.lon], {{ icon: icon }}).addTo(map);

          const popupHtml = `
            <div style="font-family: 'Inter', sans-serif; font-size: 12px; color: #0f172a; min-width: 200px;">
              <div style="font-size: 13px; font-weight: 700; color: ${{color}}; margin-bottom: 4px;">${{det.class_name}} (${{det.subtype}})</div>
              <div><b>Description:</b> ${{det.description}}</div>
              <div><b>Status:</b> ${{det.status}} (${{det.sightings}} sightings)</div>
              <div><b>Preview Z:</b> ${{det.z_m}} m (TTI: ${{det.tti_ms}} ms)</div>
              <div><b>Coordinates:</b> ${{det.lat.toFixed(6)}}, ${{det.lon.toFixed(6)}}</div>
              <div style="margin-top: 10px; border-top: 1px solid #cbd5e1; padding-top: 6px; display: flex; flex-direction: column; gap: 4px;">
                <a href="${{det.google_maps_url}}" target="_blank" style="color: #2563eb; font-weight: 600; text-decoration: none;">📍 Open in Google Maps ↗</a>
                <a href="https://www.google.com/maps/dir/?api=1&destination=${{det.lat}},${{det.lon}}" target="_blank" style="color: #059669; font-weight: 600; text-decoration: none;">🧭 Turn-by-Turn Navigation ↗</a>
              </div>
            </div>
          `;
          marker.bindPopup(popupHtml);
          hazardMarkers[hid] = marker;
        }}
      }});

      // 4. Update HUD panel telemetry
      document.getElementById('hudStep').textContent = `Step ${{idx + 1}}/${{timeline.length}}`;
      document.getElementById('hudSpeed').textContent = `${{frame.speed_kmh.toFixed(1)}} km/h`;
      document.getElementById('hudDistance').textContent = frame.nearest_z_m ? `${{frame.nearest_z_m}} m` : '-- m';
      document.getElementById('hudTti').textContent = frame.nearest_tti_ms ? `${{frame.nearest_tti_ms}} ms` : '-- ms';
      document.getElementById('hudHazardCount').textContent = Object.keys(hazardMarkers).length;

      const suspBox = document.getElementById('hudSuspensionBox');
      suspBox.className = `susp-alert-box susp-${{frame.suspension_cmd}}`;
      suspBox.textContent = `ACTIVE SUSPENSION: ${{frame.suspension_cmd}}`;

      document.getElementById('timelineSlider').value = idx;
    }}

    function playLoop() {{
      if (!isPlaying) return;
      if (currentIndex < timeline.length - 1) {{
        updateSimulationFrame(currentIndex + 1);
        const delay = 100 / playSpeed;
        timer = setTimeout(playLoop, delay);
      }} else {{
        isPlaying = false;
        updatePlayIcon();
      }}
    }}

    function togglePlayPause() {{
      isPlaying = !isPlaying;
      updatePlayIcon();
      if (isPlaying) {{
        if (currentIndex >= timeline.length - 1) currentIndex = 0;
        playLoop();
      }} else {{
        if (timer) clearTimeout(timer);
      }}
    }}

    function updatePlayIcon() {{
      const icon = document.getElementById('playIcon');
      if (isPlaying) {{
        icon.innerHTML = `<rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect>`;
      }} else {{
        icon.innerHTML = `<polygon points="5 3 19 12 5 21 5 3"></polygon>`;
      }}
    }}

    function stepForward() {{
      if (currentIndex < timeline.length - 1) {{
        updateSimulationFrame(currentIndex + 1);
      }}
    }}

    function stepBackward() {{
      if (currentIndex > 0) {{
        updateSimulationFrame(currentIndex - 1);
      }}
    }}

    function seekTo(val) {{
      updateSimulationFrame(parseInt(val, 10));
    }}

    function restartSimulation() {{
      if (timer) clearTimeout(timer);
      isPlaying = false;
      updatePlayIcon();
      // Clear markers
      Object.values(hazardMarkers).forEach(m => map.removeLayer(m));
      for (const k in hazardMarkers) delete hazardMarkers[k];
      updateSimulationFrame(0);
    }}

    function cycleSpeed() {{
      if (playSpeed === 1.0) playSpeed = 2.0;
      else if (playSpeed === 2.0) playSpeed = 5.0;
      else if (playSpeed === 5.0) playSpeed = 0.5;
      else playSpeed = 1.0;
      document.getElementById('btnSpeed').textContent = `${{playSpeed.toFixed(1)}}x`;
    }}

    // Initial frame
    window.addEventListener('DOMContentLoaded', () => {{
      updateSimulationFrame(0);
    }});
  </script>
</body>
</html>
"""

    out_file.write_text(html_content, encoding="utf-8")
    print(f"\n[Success] Interactive simulation map generated at:\n  👉 {out_file.resolve()}")
    print(f"[Success] Google My Maps KML exported at:\n  👉 {kml_file.resolve()}")
    print(f"[Success] GeoJSON exported at:\n  👉 {geojson_file.resolve()}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="GV-YOLO26 Synthetic Map Simulation Demo")
    parser.add_argument("--step-dist", type=float, default=2.0, help="Distance step in meters between frames")
    parser.add_argument("--out", default="results/synthetic_map_demo.html", help="Output HTML file path")
    parser.add_argument("--open", action="store_true", default=False, help="Automatically open map in browser")
    args = parser.parse_args()

    # 1. Generate interpolated route
    route = interpolate_route(SIMULATED_CORRIDOR["waypoints"], step_distance_m=args.step_dist)

    # 2. Run preview & detection simulation
    timeline, store = run_simulation(route)

    # 3. Generate interactive map
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / args.out

    map_path = generate_interactive_map_html(timeline, store, out_path)

    # 4. Open in browser if requested
    if args.open:
        print("\nOpening interactive simulation map in your default browser...")
        webbrowser.open(f"file://{map_path.resolve()}")


if __name__ == "__main__":
    main()
