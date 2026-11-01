"""Crowdsourced hazard store with spatial clustering (5m deduplication), GeoJSON export, and Google Maps KML generation."""

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gv_modules.hazard_mapper import (
    google_maps_dir_url,
    google_maps_url,
    haversine_distance_m,
    project_hazard_gps,
)


class HazardStore:
    """Manages crowdsourced road hazard records with spatial deduplication and map export."""

    def __init__(self, data_path: Optional[Path] = None, cluster_radius_m: float = 6.0):
        if data_path is None:
            data_path = Path(__file__).parent / "data" / "hazards.json"
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.cluster_radius_m = cluster_radius_m
        self.hazards: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load hazard records from disk, or initialize empty collection."""
        if self.data_path.exists():
            try:
                with open(self.data_path, "r", encoding="utf-8") as f:
                    self.hazards = json.load(f)
            except Exception:
                self.hazards = []
        else:
            self.hazards = []

    def _save(self):
        """Persist hazard records to disk atomically."""
        tmp_path = self.data_path.with_name(f"{self.data_path.name}.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.hazards, f, indent=2)
            tmp_path.replace(self.data_path)
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"[HazardStore] Error saving hazards: {e}")

    def record_hazard(
        self,
        class_name: str,
        ego_lat: float,
        ego_lon: float,
        heading_deg: float,
        dist_z_m: float,
        dist_x_m: float = 0.0,
        confidence: float = 0.80,
        ego_speed_kmh: float = 60.0,
    ) -> Dict[str, Any]:
        """Project and record a hazard detection. Merges into existing hazard if within cluster radius."""
        h_lat, h_lon = project_hazard_gps(ego_lat, ego_lon, heading_deg, dist_z_m, dist_x_m)
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # 1. Search for existing hazard of same class within cluster radius
        match = None
        min_dist = float("inf")
        for h in self.hazards:
            if h["class_name"] == class_name:
                d = haversine_distance_m(h_lat, h_lon, h["lat"], h["lon"])
                if d <= self.cluster_radius_m and d < min_dist:
                    match = h
                    min_dist = d

        # 2. Update existing hazard or insert new
        if match is not None:
            n = match["sightings"]
            # Weighted coordinate average
            match["lat"] = round((match["lat"] * n + h_lat) / (n + 1), 6)
            match["lon"] = round((match["lon"] * n + h_lon) / (n + 1), 6)
            match["sightings"] += 1
            match["confidence"] = round(max(match["confidence"], confidence), 3)
            match["last_seen"] = now_iso
            match["nearest_z_m"] = min(match["nearest_z_m"], round(dist_z_m, 1))
            match["ego_speed_kmh"] = round(ego_speed_kmh, 1)
            if match["sightings"] >= 2:
                match["status"] = "VERIFIED"
            match["google_maps_url"] = google_maps_url(match["lat"], match["lon"])
            res = match
        else:
            hid = f"h_{int(time.time() * 1000) % 10000000}_{len(self.hazards) + 1}"
            new_record = {
                "id": hid,
                "class_name": class_name,
                "lat": h_lat,
                "lon": h_lon,
                "confidence": round(confidence, 3),
                "sightings": 1,
                "status": "CANDIDATE",
                "first_seen": now_iso,
                "last_seen": now_iso,
                "nearest_z_m": round(dist_z_m, 1),
                "ego_speed_kmh": round(ego_speed_kmh, 1),
                "google_maps_url": google_maps_url(h_lat, h_lon),
            }
            self.hazards.append(new_record)
            res = new_record

        self._save()
        return res

    def get_all(self, class_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all recorded hazards, optionally filtered by class."""
        if class_name:
            return [h for h in self.hazards if h["class_name"] == class_name]
        return list(self.hazards)

    def clear(self):
        """Clear all stored hazards."""
        self.hazards.clear()
        self._save()

    def to_geojson(self) -> Dict[str, Any]:
        """Convert all hazards to RFC 7946 standard GeoJSON FeatureCollection."""
        features = []
        for h in self.hazards:
            feat = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    # GeoJSON is [longitude, latitude]
                    "coordinates": [h["lon"], h["lat"]],
                },
                "properties": {
                    "id": h["id"],
                    "class_name": h["class_name"],
                    "status": h["status"],
                    "sightings": h["sightings"],
                    "confidence": h["confidence"],
                    "first_seen": h["first_seen"],
                    "last_seen": h["last_seen"],
                    "nearest_z_m": h["nearest_z_m"],
                    "ego_speed_kmh": h["ego_speed_kmh"],
                    "google_maps_url": h["google_maps_url"],
                    "google_maps_dir_url": google_maps_dir_url(h["lat"], h["lon"]),
                },
            }
            features.append(feat)

        return {
            "type": "FeatureCollection",
            "metadata": {
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_hazards": len(self.hazards),
                "verified_hazards": sum(1 for h in self.hazards if h["status"] == "VERIFIED"),
                "speed_bumps": sum(1 for h in self.hazards if h["class_name"] == "Speed Bump"),
                "potholes": sum(1 for h in self.hazards if h["class_name"] == "Pothole"),
            },
            "features": features,
        }

    def to_kml(self) -> str:
        """Generate Google My Maps / Google Earth compatible KML string."""
        kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>GV-YOLO26 Road Hazard Map</name>
    <description>Crowdsourced Speed Bumps and Potholes detected by GV-YOLO26 Active Preview</description>

    <!-- Speed Bump Pin Style (Amber/Yellow) -->
    <Style id="speedBumpVerified">
      <IconStyle>
        <scale>1.2</scale>
        <Icon>
          <href>https://maps.google.com/mapfiles/ms/icons/yellow-dot.png</href>
        </Icon>
      </IconStyle>
    </Style>

    <Style id="speedBumpCandidate">
      <IconStyle>
        <scale>1.0</scale>
        <Icon>
          <href>https://maps.google.com/mapfiles/ms/icons/orange-dot.png</href>
        </Icon>
      </IconStyle>
    </Style>

    <!-- Pothole Pin Style (Red) -->
    <Style id="potholeStyle">
      <IconStyle>
        <scale>1.2</scale>
        <Icon>
          <href>https://maps.google.com/mapfiles/ms/icons/red-dot.png</href>
        </Icon>
      </IconStyle>
    </Style>
"""
        placemarks = []
        for h in self.hazards:
            is_bump = h["class_name"] == "Speed Bump"
            if is_bump:
                style_id = "#speedBumpVerified" if h["status"] == "VERIFIED" else "#speedBumpCandidate"
            else:
                style_id = "#potholeStyle"

            desc = (
                f"<![CDATA["
                f"<b>Hazard Type:</b> {h['class_name']}<br/>"
                f"<b>Status:</b> {h['status']} ({h['sightings']} sightings)<br/>"
                f"<b>Confidence:</b> {int(h['confidence'] * 100)}%<br/>"
                f"<b>Distance Z:</b> {h['nearest_z_m']} m<br/>"
                f"<b>Last Sighting:</b> {h['last_seen']}<br/>"
                f'<a href="{h["google_maps_url"]}" target="_blank">Open in Google Maps</a>'
                f"]]>"
            )

            pm = f"""    <Placemark>
      <name>{h['class_name']} ({h['status']})</name>
      <description>{desc}</description>
      <styleUrl>{style_id}</styleUrl>
      <Point>
        <coordinates>{h['lon']},{h['lat']},0</coordinates>
      </Point>
    </Placemark>"""
            placemarks.append(pm)

        kml_footer = """  </Document>
</kml>"""
        return kml_header + "\n" + "\n".join(placemarks) + "\n" + kml_footer
