#!/usr/bin/env python3
"""Launcher script for GV-YOLO26 Mobile HUD demo."""
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "demo_app"))
sys.path.insert(0, str(ROOT / "GV-YOLO26"))


def get_local_ips():
    """Retrieve non-loopback local network IP addresses."""
    ips = []
    try:
        # Connect to an external address to find default interface
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        primary = s.getsockname()[0]
        s.close()
        if primary and primary != "127.0.0.1":
            ips.append(primary)
    except Exception:
        pass

    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips or ["127.0.0.1"]


def main():
    import uvicorn

    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])

    ips = get_local_ips()

    print("\n" + "=" * 68)
    print("  🚗  ROAD HAZARD DETECTION HUD — MOBILE DEMO")
    print("=" * 68)
    print("  ⚡ Model 1 : GV-YOLO26 (2.08M Params, 5.3 GFLOPs, 82.8% mAP Backbone-Mod)")
    print("  ⭐ Model 2 : GV-YOLO26-Opt (2.08M Params, 5.8 GFLOPs, 84.1% mAP Ghost+P2)")
    print("  🏆 Model 3 : SBP-YOLO  (2.62M Params, 6.6 GFLOPs, 86.9% mAP Benchmark)")
    print("  Classes   : Speed Bump, Pothole")
    print("  Tracker   : ByteTrack with 1-2 Frame Temporal Confirmation")
    print("-" * 68)
    print("  📱 OPEN ON YOUR PHONE BROWSER:")
    for ip in ips:
        print(f"     👉  http://{ip}:{port}")
    print("\n  💻 OPEN ON YOUR LAPTOP:")
    print(f"     👉  http://localhost:{port}")
    print("=" * 68)
    print("  [Tip] Toggle between GV-YOLO26 and SBP-YOLO with 1-tap on the HUD.")
    print("  [Tip] Ensure your phone is connected to the same Wi-Fi / Hotspot.")
    print("  [Tip] Select 'Benchmark Clips' for pre-rendered evaluation playback.")
    print("  Press Ctrl+C to stop the server.\n")

    uvicorn.run("demo_app.app:app", host="0.0.0.0", port=port, reload=False, log_level="info")


if __name__ == "__main__":
    main()

