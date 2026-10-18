# GV-YOLO26 & SBP-YOLO Mobile HUD Demo Guide

This application provides a real-time, mobile-first Heads-Up Display (HUD) for demonstrating **GV-YOLO26** (proposed lightweight model) and **SBP-YOLO** (benchmark distilled model by Liang et al.) on any smartphone (iOS or Android) or laptop.

---

## 🚀 How to Run the Demo

### Step 1: Connect Phone & Laptop to the Same Network
- Connect your phone and laptop to the **same Wi-Fi network**, **OR**
- Turn on your laptop or phone's **Wi-Fi Hotspot** and connect both devices.

### Step 2: Start the Demo Server on Your Laptop
Activate the virtual environment and start the launcher:

```bash
# 1. Activate the environment (symlinked at project root)
source venv/bin/activate

# 2. Run the demo server
python run_demo.py
```

*(Alternative direct execution with mise: `mise exec -- python run_demo.py`)*

The terminal will print the exact local network URLs:
```
====================================================================
  🚗  ROAD HAZARD DETECTION HUD — DUAL-MODEL MOBILE DEMO
====================================================================
  ⚡ Model 1 : GV-YOLO26 (1.57M Params, 5.1 GFLOPs, NMS-Free Proposed)
  🏆 Model 2 : SBP-YOLO  (2.62M Params, 6.6 GFLOPs, 86.9% mAP Benchmark)
  Classes   : Speed Bump, Pothole
  Tracker   : ByteTrack with 3-Frame Temporal Confirmation
--------------------------------------------------------------------
  📱 OPEN ON YOUR PHONE BROWSER:
     👉  http://192.168.x.x:8000

  💻 OPEN ON YOUR LAPTOP:
     👉  http://localhost:8000
====================================================================
```

### Step 3: Open the URL on Your Phone Browser
Open **Safari** (iOS) or **Chrome** (Android) on your phone and navigate to the `http://192.168.x.x:8000` address displayed in your terminal.

---

## ⚡ Model Switcher (GV-YOLO26 vs. SBP-YOLO)

At the top of the HUD, you can toggle between the two models with a single tap:

1. **⚡ GV-YOLO26 (Proposed)**:
   - **Parameters**: 1.57 M (-33.7% reduction vs base YOLO26n)
   - **GFLOPs**: 5.1
   - **Architecture**: YOLO26n + VoVGSCSPC + GhostConv + LGDH
   - **Inference**: End-to-end NMS-Free Head

2. **🏆 SBP-YOLO (Benchmark)**:
   - **Parameters**: 2.62 M
   - **GFLOPs**: 6.6
   - **Accuracy**: 86.9% mAP@0.5 (Distilled student model)
   - **Architecture**: YOLO11n + VoVGSCSPC + GhostConv + LEDH (Detect_Efficient)

When switched, the tracker automatically reinitializes, and the HUD banner updates the active telemetry and model badge in real time.

---

## 📱 How to Present the Demo (3 Modes)

### Mode 1: 📹 Live Camera HUD (Real-Time Road Preview)
1. Tap the **"Start Camera"** button.
2. Grant camera permissions when prompted. The app defaults to your phone's **rear road-facing camera**.
3. Use the **"Flip Camera"** button to toggle between front and rear cameras.
4. Point your phone at road videos, pictures on your computer monitor, or real roads.
5. Watch the real-time HUD detect and track hazards:
   - 🟢 **ROAD CLEAR**: Normal driving condition.
   - 🔴 **WARNING: SPEED BUMP AHEAD**: Flashing banner + alert chime once confirmed over $\ge 3$ frames.
   - 🟠 **WARNING: POTHOLE AHEAD**: Flashing banner + alert chime.

### Mode 2: 🎬 Benchmark Presets (1-Tap Viva Presentation)
*Ideal for indoor presentations where you don't have an active road outside.*
1. Tap the **"Benchmark Clips"** tab.
2. Tap any of the 4 test scenes from the project report:
   - **Clip 1: Daytime Rain** *(Rain, wet road, painted speed bumps)*
   - **Clip 2: Night Scene** *(Well-marked speed bumps under vehicle headlights)*
   - **Clip 3: Night Dashcam** *(Urban nighttime road hazards)*
   - **Clip 4: Dusk Scene** *(Low-contrast dusk bumps and potholes)*
3. The video plays with pre-computed bounding boxes, ByteTrack IDs, and HUD warnings.

### Mode 3: 📁 Upload Media / Snap Photo
- **Snap Photo**: Take a quick photo directly from your camera to run single-frame high-resolution inference.
- **Select Video**: Upload any test dashcam video file for server-side processing.

---

## ⚙️ Technical Architecture
- **Backend**: FastAPI with async image streaming, OpenCV HUD annotation, and ByteTrack tracker.
- **Detection Models**: Ultralytics YOLO with custom modules (`gv_modules` for GV-YOLO26 and `sbp_modules` for SBP-YOLO).
- **Frontend**: Single-page responsive HTML5/CSS3 with Web Audio API synthesizer chimes and zero external dependencies.
