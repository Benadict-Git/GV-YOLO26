#!/usr/bin/env python3
"""
Generate a publication-grade System Workflow diagram for GV-YOLO26
styled authentically as if drawn in draw.io.
"""

import html
import os
import subprocess

def esc(s):
    if not isinstance(s, str):
        return str(s)
    return html.escape(s, quote=False)

def create_drawio_system_workflow_svg(output_svg_path):
    # Canvas dimensions
    W = 1440
    H = 1580
    
    # Draw.io authentic color palette (Classic Draw.io UI themes)
    # Phase 1: Blue / Cyan (Data Pipeline)
    C1_BG = "#F4F8FD"
    C1_BORDER = "#6C8EBF"
    C1_HDR_BG = "#DAE8FC"
    C1_HDR_TXT = "#1A365D"
    C1_ACCENT = "#2563EB"
    
    # Phase 2: Purple / Indigo (Architecture)
    C2_BG = "#FAF7FD"
    C2_BORDER = "#9673A6"
    C2_HDR_BG = "#E1D5E7"
    C2_HDR_TXT = "#3B1A54"
    C2_ACCENT = "#7C3AED"
    
    # Phase 3: Amber / Orange (Training & Loss)
    C3_BG = "#FEFBF4"
    C3_BORDER = "#D79B00"
    C3_HDR_BG = "#FFE6CC"
    C3_HDR_TXT = "#6B3800"
    C3_ACCENT = "#D97706"
    
    # Phase 4: Green / Emerald (Deployment & Telemetry)
    C4_BG = "#F5FAF6"
    C4_BORDER = "#82B366"
    C4_HDR_BG = "#D5E8D4"
    C4_HDR_TXT = "#1E4620"
    C4_ACCENT = "#059669"
    
    # Card styles
    CARD_BG = "#FFFFFF"
    CARD_STROKE = "#CBD5E1"
    
    # Text colors
    TXT_DARK = "#1E293B"
    TXT_MUTED = "#475569"
    TXT_SUBTLE = "#64748B"
    
    # Connector colors
    CONN_STROKE = "#334155"
    BUS_STROKE = "#1E3A8A"

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    
    # Definitions: Filters, markers, shadows
    svg.append('''
    <defs>
      <!-- Draw.io subtle drop shadow -->
      <filter id="drawio-shadow" x="-5%" y="-5%" width="112%" height="114%" filterUnits="userSpaceOnUse">
        <feGaussianBlur in="SourceAlpha" stdDeviation="2.5" />
        <feOffset dx="1.5" dy="2.5" />
        <feComponentTransfer><feFuncA type="linear" slope="0.12"/></feComponentTransfer>
        <feMerge>
          <feMergeNode />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
      
      <!-- Container subtle shadow -->
      <filter id="container-shadow" x="-2%" y="-2%" width="105%" height="106%" filterUnits="userSpaceOnUse">
        <feGaussianBlur in="SourceAlpha" stdDeviation="3" />
        <feOffset dx="0" dy="2" />
        <feComponentTransfer><feFuncA type="linear" slope="0.08"/></feComponentTransfer>
        <feMerge>
          <feMergeNode />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>

      <!-- Draw.io arrowheads -->
      <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155" />
      </marker>
      
      <marker id="bus-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#1E3A8A" />
      </marker>
    </defs>
    
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .title { font-size: 24px; font-weight: 800; fill: #0F172A; letter-spacing: -0.5px; }
      .subtitle { font-size: 13px; font-weight: 500; fill: #475569; }
      .phase-hdr-title { font-size: 14px; font-weight: 700; letter-spacing: 0.3px; dominant-baseline: central; }
      .phase-hdr-tag { font-size: 11px; font-weight: 600; dominant-baseline: central; }
      .card-title { font-size: 13.5px; font-weight: 700; fill: #0F172A; }
      .code-badge { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 10px; font-weight: 600; dominant-baseline: central; }
      .body-text { font-size: 11px; fill: #334155; line-height: 1.45; }
      .bold-text { font-weight: 600; fill: #0F172A; }
      .tag-text { font-size: 9.5px; font-weight: 600; dominant-baseline: central; }
      .bus-label { font-size: 10.5px; font-weight: 600; fill: #1E3A8A; dominant-baseline: central; text-anchor: middle; }
    </style>
    ''')
    
    # Canvas Background
    svg.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF" />')
    
    # Outer subtle boundary grid / border (Draw.io canvas border)
    svg.append(f'<rect x="15" y="15" width="{W-30}" height="{H-30}" rx="12" fill="none" stroke="#E2E8F0" stroke-width="1.5" />')
    
    # Title Header Block (Draw.io style banner)
    svg.append(f'''
    <g transform="translate(45, 35)">
      <!-- Main Header pill -->
      <rect x="0" y="0" width="1350" height="70" rx="8" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="1.5" filter="url(#container-shadow)" />
      <rect x="0" y="0" width="6" height="70" rx="3" fill="#2563EB" />
      
      <!-- Text -->
      <text x="24" y="32" class="title">GV-YOLO26: End-to-End System Workflow Architecture</text>
      <text x="24" y="54" class="subtitle">From Raw SBP-YOLO Dataset Preprocessing &amp; Architecture Engineering to Embedded TensorRT Preview Control</text>
      
      <!-- Meta badges on top right -->
      <g transform="translate(980, 18)">
        <rect x="0" y="0" width="115" height="34" rx="6" fill="#EFF6FF" stroke="#BFDBFE" stroke-width="1" />
        <text x="12" y="14" class="code-badge" fill="#1D4ED8">INPUT: 640×640</text>
        <text x="12" y="27" font-size="9px" fill="#3B82F6">2 Hazard Classes</text>
        
        <rect x="125" y="0" width="115" height="34" rx="6" fill="#F0FDF4" stroke="#BBF7D0" stroke-width="1" />
        <text x="137" y="14" class="code-badge" fill="#15803D">SPEED: 172 FPS</text>
        <text x="137" y="27" font-size="9px" fill="#16A34A">5.8 ms Latency</text>
        
        <rect x="250" y="0" width="105" height="34" rx="6" fill="#FEF3C7" stroke="#FDE68A" stroke-width="1" />
        <text x="260" y="14" class="code-badge" fill="#B45309">ACC: 84.97%</text>
        <text x="260" y="27" font-size="9px" fill="#D97706">Held-out mAP50</text>
      </g>
    </g>
    ''')

    # Detailed Phase & Card Configurations
    phases = [
        {
            "num": "1",
            "title": "PHASE 1: DATA CURATION, LEAKAGE FILTERING & PREPROCESSING",
            "tag": "MODULE: dataset_prep & gv_modules/augment.py",
            "bg": C1_BG, "border": C1_BORDER, "hdr_bg": C1_HDR_BG, "hdr_txt": C1_HDR_TXT, "accent": C1_ACCENT,
            "y": 125, "h": 285,
            "cards": [
                {
                    "type": "cylinder",
                    "title": "Raw SBP-YOLO Dataset",
                    "badge": "Kaggle speed12",
                    "sub": "Raw Distribution (6,954 images)",
                    "bullets": [
                        ("• Total Images: ", "6,954 annotated frames"),
                        ("• Potholes: ", "12,205 annotations (Class 0)"),
                        ("• Speed Bumps: ", "3,217 annotations (Class 1)"),
                        ("• Raw Partitions: ", "train/ (5,829), valid/ (586), test/ (1,125)")
                    ],
                    "tag_box": "Contains spatio-temporal leakage across folders",
                    "tag_color": "#DC2626", "tag_bg": "#FEF2F2", "tag_border": "#FCA5A5"
                },
                {
                    "type": "rect",
                    "title": "Leakage-Free Partitioning",
                    "badge": "tools/make_clean_split.py",
                    "sub": "De-duplication & Video Sequence Filtering",
                    "bullets": [
                        ("• Frame De-duplication: ", "Identifies augmented duplicates"),
                        ("• Zero-Leakage Holdout: ", "482 images strictly isolated"),
                        ("• Carved Validation Split: ", "586 images from train/ (seed 42)"),
                        ("• Training Partition: ", "5,243 clean images for training")
                    ],
                    "tag_box": "Guarantees unbiased empirical metric validation",
                    "tag_color": "#16A34A", "tag_bg": "#F0FDF4", "tag_border": "#86EFAC"
                },
                {
                    "type": "rect",
                    "title": "Atmospheric Simulation",
                    "badge": "gv_modules/augment.py",
                    "sub": "Adverse Weather & Domain Augmentation",
                    "bullets": [
                        ("• Weather Synthesizer: ", "Rain streaks, snow noise (Albumentations)"),
                        ("• Optical Variations: ", "Motion blur, glare & shadow shifts"),
                        ("• Spatial Normalization: ", "Aspect-preserved letterbox resize"),
                        ("• Output Tensor: ", "640 × 640 × 3 normalized tensors")
                    ],
                    "tag_box": "Prevents environmental overfitting in real driving",
                    "tag_color": "#2563EB", "tag_bg": "#EFF6FF", "tag_border": "#BFDBFE"
                }
            ]
        },
        {
            "num": "2",
            "title": "PHASE 2: GV-YOLO26 ARCHITECTURAL TOPOLOGY & LAYER FORMULATION",
            "tag": "MODULE: gv_modules/blocks.py, head.py & models/gv-yolo26.yaml",
            "bg": C2_BG, "border": C2_BORDER, "hdr_bg": C2_HDR_BG, "hdr_txt": C2_HDR_TXT, "accent": C2_ACCENT,
            "y": 465, "h": 285,
            "cards": [
                {
                    "type": "rect",
                    "title": "Backbone & High-Res P2 Branch",
                    "badge": "models/gv-yolo26.yaml",
                    "sub": "High-Resolution Spatial Detail Recovery",
                    "bullets": [
                        ("• Base Network: ", "YOLO26n stem + C3k2 residual stages"),
                        ("• P2 Feature Branch: ", "Stride 4 level with 160 × 160 grid"),
                        ("• Resolution Detail: ", "4 cells on 16-px distant speed bumps"),
                        ("• Small Target Capture: ", "+2.7% mAP50 improvement on potholes")
                    ],
                    "tag_box": "Eliminates small-object information loss",
                    "tag_color": "#7C3AED", "tag_bg": "#FAF5FF", "tag_border": "#DDD6FE"
                },
                {
                    "type": "rect",
                    "title": "Lightweight Spatial Mixing",
                    "badge": "gv_modules/blocks.py",
                    "sub": "GhostConv & VoVGSCSPC Optimization",
                    "bullets": [
                        ("• GhostConv Downsampling: ", "Cheap linear transforms halve FLOPs"),
                        ("• VoVGSCSPC Neck: ", "One-Shot Aggregation cross-stage fusion"),
                        ("• Parameter Compression: ", "Drops backbone params by 0.23M"),
                        ("• Dynamic Receptive Field: ", "SPPF + C2PSA global context attention")
                    ],
                    "tag_box": "Balances spatial richness with edge compute budget",
                    "tag_color": "#9333EA", "tag_bg": "#FAF5FF", "tag_border": "#E9D5FF"
                },
                {
                    "type": "rect",
                    "title": "Lightweight Grouped Detection Head",
                    "badge": "gv_modules/head.py (LGDH)",
                    "sub": "Shared Grouped Convolutions (NMS-Free)",
                    "bullets": [
                        ("• Grouped Sharing: ", "c/8 groups (channels_per_group=8)"),
                        ("• FLOP Control: ", "P2 head cost slashed from 6.6 to 6.2 GFLOPs"),
                        ("• No DFL Overhead: ", "Direct 4-channel box regression"),
                        ("• Dual Architecture: ", "Auxiliary 1-to-many train; 1-to-1 deploy")
                    ],
                    "tag_box": "Zero post-processing NMS latency in inference",
                    "tag_color": "#4F46E5", "tag_bg": "#EEF2FF", "tag_border": "#C7D2FE"
                }
            ]
        },
        {
            "num": "3",
            "title": "PHASE 3: LOSS OPTIMIZATION, DISTILLATION & PROGRESSIVE ABLATION",
            "tag": "MODULE: gv_modules/loss.py, distill.py & train.py",
            "bg": C3_BG, "border": C3_BORDER, "hdr_bg": C3_HDR_BG, "hdr_txt": C3_HDR_TXT, "accent": C3_ACCENT,
            "y": 805, "h": 285,
            "cards": [
                {
                    "type": "rect",
                    "title": "Normalized Wasserstein Distance Loss",
                    "badge": "gv_modules/loss.py",
                    "sub": "Gaussian Distribution Regression Metric",
                    "bullets": [
                        ("• Metric Paradigm: ", "Models bounding boxes as 2D Gaussians"),
                        ("• Non-Overlap Gradient: ", "Smooth distance even when IoU = 0"),
                        ("• Combined Formulation: ", "L_box = L_CIoU + λ_nwd · L_NWD"),
                        ("• Regression Stability: ", "Prevents gradient collapse on tiny boxes")
                    ],
                    "tag_box": "Overcomes IoU extreme sensitivity on micro-potholes",
                    "tag_color": "#D97706", "tag_bg": "#FFFBEB", "tag_border": "#FDE68A"
                },
                {
                    "type": "rect",
                    "title": "Teacher-Student Distillation",
                    "badge": "distill.py (BCKD + CWD)",
                    "sub": "Cross-Scale Feature & Box Alignment",
                    "bullets": [
                        ("• Teacher Model: ", "GV-YOLO26-S (Small-scale teacher)"),
                        ("• Channel Distillation: ", "CWD mimics intermediate feature maps"),
                        ("• Box Distillation: ", "BCKD student-teacher box alignment"),
                        ("• Softplus Clamping: ", "Prevents negative GIoU gradient divergence")
                    ],
                    "tag_box": "Explores teacher knowledge transfer mechanisms",
                    "tag_color": "#EA580C", "tag_bg": "#FFF7ED", "tag_border": "#FFEDD5"
                },
                {
                    "type": "rect",
                    "title": "Progressive 6-Stage Ablation Ladder",
                    "badge": "train.py --nwd --cfg ...",
                    "sub": "Empirical Verification Across Stages",
                    "bullets": [
                        ("• S1 YOLO26n Base: ", "75.5% mAP50 | 5.3 GFLOPs | 2.59M"),
                        ("• S2 (+P2 branch): ", "78.2% mAP50 | 6.6 GFLOPs | 2.81M"),
                        ("• S3-S4 (+Ghost+VoV): ", "80.0% mAP50 | 6.5 GFLOPs | 2.58M"),
                        ("• S5 (+LGDH Head): ", "80.8% mAP50 | 6.2 GFLOPs | 2.58M"),
                        ("• S6 (+NWD Loss): ", "84.97% mAP50 (Best Checkpoint)")
                    ],
                    "tag_box": "+9.5% mAP50 gain over YOLO26n at only 6.2 GFLOPs",
                    "tag_color": "#B45309", "tag_bg": "#FEF3C7", "tag_border": "#FCD34D"
                }
            ]
        },
        {
            "num": "4",
            "title": "PHASE 4: EMBEDDED OPTIMIZATION, TRACKING & ACTIVE PREVIEW CONTROL",
            "tag": "MODULE: export_trt.py, demo_video.py & demo_app/app.py",
            "bg": C4_BG, "border": C4_BORDER, "hdr_bg": C4_HDR_BG, "hdr_txt": C4_HDR_TXT, "accent": C4_ACCENT,
            "y": 1145, "h": 285,
            "cards": [
                {
                    "type": "chip",
                    "title": "Embedded TensorRT Acceleration",
                    "badge": "export_trt.py --half",
                    "sub": "Hardware Compilation (NVIDIA Jetson)",
                    "bullets": [
                        ("• Target Hardware: ", "NVIDIA Jetson AGX Xavier / Orin"),
                        ("• Quantization Mode: ", "FP16 half-precision & INT8 calibration"),
                        ("• Kernel Optimization: ", "End-to-end fused engine (no NMS node)"),
                        ("• Benchmarked Latency: ", "5.8 ms / frame (172+ FPS edge execution)")
                    ],
                    "tag_box": "11.6× faster than the 15 Hz suspension reaction limit",
                    "tag_color": "#059669", "tag_bg": "#F0FDF4", "tag_border": "#A7F3D0"
                },
                {
                    "type": "rect",
                    "title": "ByteTrack Multi-Object Association",
                    "badge": "demo_video.py --track",
                    "sub": "Temporal Trajectory Filtering",
                    "bullets": [
                        ("• Track Association: ", "ByteTrack Kalman filtering across frames"),
                        ("• Spatial Trajectory: ", "Links bounding boxes over vehicle approach"),
                        ("• 3-Frame Confirmation: ", "Suppresses single-frame road false positives"),
                        ("• Hazard Tracking: ", "Persistent ID maintenance until passover")
                    ],
                    "tag_box": "Eliminates transient road surface visual noise",
                    "tag_color": "#047857", "tag_bg": "#ECFDF5", "tag_border": "#6EE7B7"
                },
                {
                    "type": "car",
                    "title": "Active Suspension & HUD Telemetry",
                    "badge": "demo_app/ (FastAPI + WebSocket)",
                    "sub": "Vehicular Preview Control Interface",
                    "bullets": [
                        ("• Actuator Preview: ", "CAN bus signal softens dampers before impact"),
                        ("• Distance Profiling: ", "Relative distance & severity estimation"),
                        ("• Driver Warning HUD: ", "Real-time visual banner + audio buzzer alert"),
                        ("• Dual Architecture: ", "Side-by-side benchmark mode vs SBP-YOLO")
                    ],
                    "tag_box": "Mitigates vertical cabin shock & chassis strain",
                    "tag_color": "#15803D", "tag_bg": "#F0FDF4", "tag_border": "#86EFAC"
                }
            ]
        }
    ]

    # Render each phase container and its 3 cards
    for p_idx, p in enumerate(phases):
        py = p["y"]
        ph = p["h"]
        
        # Outer container
        svg.append(f'''
        <g id="phase-{p['num']}" transform="translate(45, {py})">
          <!-- Main Container Box with draw.io style -->
          <rect x="0" y="0" width="1350" height="{ph}" rx="8" fill="{p['bg']}" stroke="{p['border']}" stroke-width="1.8" filter="url(#container-shadow)" />
          
          <!-- Header Tab / Bar -->
          <path d="M 0 8 A 8 8 0 0 1 8 0 L 1342 0 A 8 8 0 0 1 1350 8 L 1350 36 L 0 36 Z" fill="{p['hdr_bg']}" stroke="{p['border']}" stroke-width="1.8" />
          <line x1="0" y1="36" x2="1350" y2="36" stroke="{p['border']}" stroke-width="1.8" />
          
          <!-- Phase Indicator Pill -->
          <rect x="14" y="7" width="24" height="22" rx="4" fill="{p['accent']}" />
          <text x="21" y="19" font-size="13px" font-weight="800" fill="#FFFFFF" dominant-baseline="central">{p['num']}</text>
          
          <!-- Header Text -->
          <text x="48" y="19" class="phase-hdr-title" fill="{p['hdr_txt']}">{esc(p['title'])}</text>
          
          <!-- Header Subtitle / Tag on Right -->
          <rect x="920" y="7" width="415" height="22" rx="4" fill="#FFFFFF" stroke="{p['border']}" stroke-width="1" />
          <text x="932" y="19" class="phase-hdr-tag" fill="{p['hdr_txt']}">{esc(p['tag'])}</text>
        ''')
        
        # Render the 3 cards inside
        card_w = 405
        card_h = 224
        card_y = 48
        card_gap = 45
        card_x_start = 22
        
        for c_idx, c in enumerate(p["cards"]):
            cx = card_x_start + c_idx * (card_w + card_gap)
            
            # Card background
            svg.append(f'''
            <g transform="translate({cx}, {card_y})">
              <!-- Draw.io Card Rectangle -->
              <rect x="0" y="0" width="{card_w}" height="{card_h}" rx="6" fill="{CARD_BG}" stroke="{CARD_STROKE}" stroke-width="1.5" filter="url(#drawio-shadow)" />
              
              <!-- Subtle top highlight line matching phase accent -->
              <path d="M 0 6 A 6 6 0 0 1 6 0 L {card_w-6} 0 A 6 6 0 0 1 {card_w} 6 L {card_w} 9 L 0 9 Z" fill="{p['accent']}" />
            ''')
            
            # Draw.io Icons on Top Right of Card
            if c["type"] == "cylinder":
                # Iconic Draw.io Database Cylinder
                svg.append(f'''
                <g transform="translate({card_w-40}, 15)">
                  <path d="M 0 6 L 0 18 A 13 5 0 0 0 26 18 L 26 6 Z" fill="#EEF2FF" stroke="#4F46E5" stroke-width="1.3" />
                  <ellipse cx="13" cy="6" rx="13" ry="4.5" fill="#E0E7FF" stroke="#4F46E5" stroke-width="1.3" />
                  <path d="M 0 12 A 13 4.5 0 0 0 26 12" fill="none" stroke="#4F46E5" stroke-width="1" />
                </g>
                ''')
            elif c["type"] == "chip":
                # Microchip / TensorRT Processor Icon
                svg.append(f'''
                <g transform="translate({card_w-38}, 16)">
                  <rect x="4" y="4" width="18" height="18" rx="3" fill="#ECFDF5" stroke="#059669" stroke-width="1.3" />
                  <rect x="8" y="8" width="10" height="10" rx="1.5" fill="#D1FAE5" />
                  <line x1="1" y1="9" x2="4" y2="9" stroke="#059669" stroke-width="1.2" />
                  <line x1="1" y1="17" x2="4" y2="17" stroke="#059669" stroke-width="1.2" />
                  <line x1="22" y1="9" x2="25" y2="9" stroke="#059669" stroke-width="1.2" />
                  <line x1="22" y1="17" x2="25" y2="17" stroke="#059669" stroke-width="1.2" />
                </g>
                ''')
            elif c["type"] == "car":
                # Vehicle / Active Damper Icon
                svg.append(f'''
                <g transform="translate({card_w-42}, 16)">
                  <path d="M 2 15 L 6 8 L 22 8 L 26 15 Z" fill="#DCFCE7" stroke="#15803D" stroke-width="1.3" />
                  <rect x="0" y="15" width="28" height="6" rx="2" fill="#BBF7D0" stroke="#15803D" stroke-width="1.2" />
                  <circle cx="6" cy="21" r="3" fill="#15803D" />
                  <circle cx="22" cy="21" r="3" fill="#15803D" />
                </g>
                ''')
            
            # Title & Badge
            badge_w = len(c['badge']) * 7 + 16
            svg.append(f'''
              <text x="14" y="28" class="card-title">{esc(c['title'])}</text>
              <rect x="14" y="36" width="{badge_w}" height="17" rx="3" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="1" />
              <text x="22" y="45.5" class="code-badge" fill="{TXT_MUTED}">{esc(c['badge'])}</text>
              <text x="14" y="70" font-size="11px" font-weight="600" fill="{TXT_SUBTLE}">{esc(c['sub'])}</text>
              
              <line x1="14" y1="78" x2="{card_w-14}" y2="78" stroke="#E2E8F0" stroke-width="1" />
            ''')
            
            # Bullets (with xml:space="preserve" to maintain spacing)
            by = 96
            for prefix, val in c["bullets"]:
                svg.append(f'''
                <text x="14" y="{by}" class="body-text" xml:space="preserve"><tspan class="bold-text">{esc(prefix)}</tspan><tspan fill="{TXT_MUTED}">{esc(val)}</tspan></text>
                ''')
                by += 19
                
            # Bottom callout pill / tag box
            svg.append(f'''
              <rect x="12" y="{card_h - 30}" width="{card_w - 24}" height="22" rx="4" fill="{c['tag_bg']}" stroke="{c['tag_border']}" stroke-width="1" />
              <text x="20" y="{card_h - 18}" class="tag-text" fill="{c['tag_color']}">{esc(c['tag_box'])}</text>
            </g>
            ''')
            
            # Horizontal connectors between cards inside the same phase
            if c_idx < 2:
                arrow_x1 = cx + card_w
                arrow_x2 = arrow_x1 + card_gap
                arrow_y = card_y + card_h // 2 - 10
                svg.append(f'''
                <!-- Intra-phase arrow from Card {c_idx+1} to {c_idx+2} -->
                <line x1="{arrow_x1}" y1="{arrow_y}" x2="{arrow_x2}" y2="{arrow_y}" stroke="{CONN_STROKE}" stroke-width="2" marker-end="url(#arrow)" />
                <circle cx="{arrow_x1}" cy="{arrow_y}" r="3" fill="{CONN_STROKE}" />
                ''')
                
        svg.append('</g>') # close phase group

    # Inter-Phase Bus Connectors (Draw.io Container-to-Container Routing)
    # The connector begins at the bottom border of Phase container N under Card C,
    # travels across the inter-phase gap through a centered label pill,
    # and enters the top border of Phase container N+1 directly above Card A!
    
    card_c_cx = 45 + 22 + 2 * 450 + 202  # 1169
    card_a_cx = 45 + 22 + 202            # 269
    
    # 1. Bus: Phase 1 (Card 1C) -> Phase 2 (Card 2A)
    p1_bot = phases[0]["y"] + phases[0]["h"]
    p2_top = phases[1]["y"]
    bus1_mid_y = (p1_bot + p2_top) // 2
    label1_w = 380
    label1_x = (W - label1_w) // 2
    
    svg.append(f'''
    <!-- Inter-Phase Connector: Phase 1 -> Phase 2 -->
    <g id="inter-phase-1-2">
      <path d="M {card_c_cx} {p1_bot} L {card_c_cx} {bus1_mid_y} L {card_a_cx} {bus1_mid_y} L {card_a_cx} {p2_top}" fill="none" stroke="{BUS_STROKE}" stroke-width="2.2" stroke-linejoin="round" marker-end="url(#bus-arrow)" />
      <circle cx="{card_c_cx}" cy="{p1_bot}" r="3.5" fill="{BUS_STROKE}" />
      
      <!-- Label Box on the bus -->
      <g transform="translate({label1_x}, {bus1_mid_y - 13})">
        <rect x="0" y="0" width="{label1_w}" height="26" rx="6" fill="#FFFFFF" stroke="#3B82F6" stroke-width="1.6" filter="url(#drawio-shadow)" />
        <text x="{label1_w // 2}" y="13" class="bus-label">Clean Preprocessed 640×640 Tensors + Zero-Leakage Split</text>
      </g>
    </g>
    ''')
    
    # 2. Bus: Phase 2 (Card 2C) -> Phase 3 (Card 3A)
    p2_bot = phases[1]["y"] + phases[1]["h"]
    p3_top = phases[2]["y"]
    bus2_mid_y = (p2_bot + p3_top) // 2
    label2_w = 420
    label2_x = (W - label2_w) // 2
    
    svg.append(f'''
    <!-- Inter-Phase Connector: Phase 2 -> Phase 3 -->
    <g id="inter-phase-2-3">
      <path d="M {card_c_cx} {p2_bot} L {card_c_cx} {bus2_mid_y} L {card_a_cx} {bus2_mid_y} L {card_a_cx} {p3_top}" fill="none" stroke="{BUS_STROKE}" stroke-width="2.2" stroke-linejoin="round" marker-end="url(#bus-arrow)" />
      <circle cx="{card_c_cx}" cy="{p2_bot}" r="3.5" fill="{BUS_STROKE}" />
      
      <!-- Label Box on the bus -->
      <g transform="translate({label2_x}, {bus2_mid_y - 13})">
        <rect x="0" y="0" width="{label2_w}" height="26" rx="6" fill="#FFFFFF" stroke="#7C3AED" stroke-width="1.6" filter="url(#drawio-shadow)" />
        <text x="{label2_w // 2}" y="13" class="bus-label" fill="#6D28D9">Configured GV-YOLO26 Model (P2 Level + Ghost-VoV Neck + LGDH)</text>
      </g>
    </g>
    ''')

    # 3. Bus: Phase 3 (Card 3C - S6 Best Checkpoint!) -> Phase 4 (Card 4A)
    p3_bot = phases[2]["y"] + phases[2]["h"]
    p4_top = phases[3]["y"]
    bus3_mid_y = (p3_bot + p4_top) // 2
    label3_w = 440
    label3_x = (W - label3_w) // 2
    
    svg.append(f'''
    <!-- Inter-Phase Connector: Phase 3 -> Phase 4 -->
    <g id="inter-phase-3-4">
      <path d="M {card_c_cx} {p3_bot} L {card_c_cx} {bus3_mid_y} L {card_a_cx} {bus3_mid_y} L {card_a_cx} {p4_top}" fill="none" stroke="{BUS_STROKE}" stroke-width="2.2" stroke-linejoin="round" marker-end="url(#bus-arrow)" />
      <circle cx="{card_c_cx}" cy="{p3_bot}" r="3.5" fill="{BUS_STROKE}" />
      
      <!-- Label Box on the bus -->
      <g transform="translate({label3_x}, {bus3_mid_y - 13})">
        <rect x="0" y="0" width="{label3_w}" height="26" rx="6" fill="#FFFFFF" stroke="#D97706" stroke-width="1.6" filter="url(#drawio-shadow)" />
        <text x="{label3_w // 2}" y="13" class="bus-label" fill="#B45309">Best Stage-6 Checkpoint: gv_yolo26_simple_best.pt (84.97% mAP50)</text>
      </g>
    </g>
    ''')

    # Bottom Footer Summary / Hardware Execution Note
    svg.append(f'''
    <g transform="translate(45, 1455)">
      <rect x="0" y="0" width="1350" height="75" rx="8" fill="#F8FAFC" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="6,4" />
      
      <!-- Icon pill -->
      <rect x="18" y="16" width="42" height="42" rx="6" fill="#0F172A" />
      <text x="27" y="42" font-size="20px" fill="#38BDF8">⚡</text>
      
      <!-- Text details -->
      <text x="74" y="32" font-size="12.5px" font-weight="700" fill="#0F172A">End-to-End Real-Time System Execution Protocol</text>
      <text x="74" y="52" font-size="11px" fill="#475569">
        Input Camera (640×640) → NMS-Free TensorRT Engine (5.8 ms) → ByteTrack Association (3-frame temporal gate) → Active Suspension Damper Softening &amp; Driver HUD Alert.
      </text>
      
      <!-- Status pill on right -->
      <g transform="translate(1080, 22)">
        <rect x="0" y="0" width="250" height="30" rx="15" fill="#DCFCE7" stroke="#86EFAC" stroke-width="1.2" />
        <circle cx="16" cy="15" r="5" fill="#16A34A" />
        <text x="28" y="19" font-size="10.5px" font-weight="700" fill="#15803D">NMS-FREE REAL-TIME CERTIFIED</text>
      </g>
    </g>
    ''')

    svg.append('</svg>')
    
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"SVG written to {output_svg_path}")

if __name__ == '__main__':
    svg_path = "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.svg"
    png_path = "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.png"
    create_drawio_system_workflow_svg(svg_path)
    
    # Render with rsvg-convert at high resolution (2880 width)
    cmd = ["rsvg-convert", "-w", "2880", "--format", "png", "-o", png_path, svg_path]
    subprocess.run(cmd, check=True)
    print(f"High-res PNG rendered to {png_path}")
