import subprocess
import os

svg_content = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="2300" viewBox="0 0 1600 2300">
  <defs>
    <filter id="card-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.08"/>
    </filter>
    <filter id="pill-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1e3a8a" flood-opacity="0.25"/>
    </filter>
    
    <!-- Gradients -->
    <linearGradient id="start-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>
    <linearGradient id="end-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#1e293b"/>
      <stop offset="100%" stop-color="#0f172a"/>
    </linearGradient>
    <linearGradient id="core-card-grad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f0fdf4"/>
    </linearGradient>

    <!-- Markers for arrows -->
    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#475569"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#059669"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="100%" height="100%" fill="#ffffff"/>

  <!-- ================= BACKGROUND PHASE CONTAINERS ================= -->
  
  <!-- Phase 1: Baseline Benchmarking (Steps 1-4) -->
  <g id="phase1">
    <rect x="180" y="160" width="1240" height="660" rx="20" fill="#f8fafc" stroke="#e2e8f0" stroke-width="2"/>
    <rect x="200" y="180" width="310" height="32" rx="8" fill="#e2e8f0"/>
    <text x="355" y="201" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="13" font-weight="bold" fill="#475569" text-anchor="middle" letter-spacing="0.5">PHASE 1: BASELINE &amp; SELECTION</text>
  </g>

  <!-- Phase 2: Core Architectural Innovation (Step 5) -->
  <g id="phase2">
    <rect x="180" y="855" width="1240" height="245" rx="20" fill="#f0fdf4" stroke="#bbf7d0" stroke-width="2"/>
    <rect x="200" y="875" width="330" height="32" rx="8" fill="#dcfce7"/>
    <text x="365" y="896" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="13" font-weight="bold" fill="#15803d" text-anchor="middle" letter-spacing="0.5">PHASE 2: ARCHITECTURAL DESIGN</text>
  </g>

  <!-- Phase 3: Progressive Training & Distillation (Steps 6-7) -->
  <g id="phase3">
    <rect x="180" y="1135" width="1240" height="420" rx="20" fill="#fffbeb" stroke="#fef3c7" stroke-width="2"/>
    <rect x="200" y="1155" width="310" height="32" rx="8" fill="#fef3c7"/>
    <text x="355" y="1176" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="13" font-weight="bold" fill="#b45309" text-anchor="middle" letter-spacing="0.5">PHASE 3: TRAINING &amp; DISTILLATION</text>
  </g>

  <!-- Phase 4: Evaluation & Real-Time Deployment (Steps 8-9) -->
  <g id="phase4">
    <rect x="180" y="1590" width="1240" height="420" rx="20" fill="#faf5ff" stroke="#f3e8ff" stroke-width="2"/>
    <rect x="200" y="1610" width="320" height="32" rx="8" fill="#f3e8ff"/>
    <text x="360" y="1631" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="13" font-weight="bold" fill="#7e22ce" text-anchor="middle" letter-spacing="0.5">PHASE 4: EVALUATION &amp; DEPLOYMENT</text>
  </g>


  <!-- ================= FLOW NODES & ARROWS ================= -->

  <!-- START Terminal Node -->
  <g id="node-start">
    <rect x="670" y="50" width="260" height="64" rx="32" fill="url(#start-grad)" filter="url(#pill-shadow)"/>
    <text x="800" y="90" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="20" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="1">Start</text>
  </g>

  <!-- Arrow: Start -> Step 1 -->
  <line x1="800" y1="114" x2="800" y2="230" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 1: Review literature -->
  <g id="step1">
    <rect x="360" y="235" width="880" height="96" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="291" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Review literature on bump and pothole detection</text>
  </g>

  <!-- Arrow: Step 1 -> Step 2 -->
  <line x1="800" y1="331" x2="800" y2="385" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 2: Train stock baselines -->
  <g id="step2">
    <rect x="360" y="390" width="880" height="96" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="446" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Train seven stock baselines on public dataset (fixed protocol)</text>
  </g>

  <!-- Arrow: Step 2 -> Step 3 -->
  <line x1="800" y1="486" x2="800" y2="540" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 3: Evaluate baselines -->
  <g id="step3">
    <rect x="360" y="545" width="880" height="96" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="601" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Evaluate baselines on held-out test split</text>
  </g>

  <!-- Arrow: Step 3 -> Step 4 -->
  <line x1="800" y1="641" x2="800" y2="695" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 4: Select YOLO26n -->
  <g id="step4">
    <rect x="360" y="700" width="880" height="96" rx="14" fill="#ffffff" stroke="#94a3b8" stroke-width="2" filter="url(#card-shadow)"/>
    <text x="800" y="743" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Select YOLO26n as base</text>
    <text x="800" y="771" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="16" font-weight="500" fill="#64748b" text-anchor="middle">(most accurate, lowest cost)</text>
  </g>

  <!-- Arrow: Step 4 -> Step 5 (Core Architecture) -->
  <line x1="800" y1="796" x2="800" y2="925" stroke="#059669" stroke-width="3" marker-end="url(#arrow-emerald)"/>

  <!-- Step 5: Design GV-YOLO26 (Highlighted Core Contribution) -->
  <g id="step5">
    <rect x="330" y="930" width="940" height="135" rx="16" fill="url(#core-card-grad)" stroke="#059669" stroke-width="2.5" filter="url(#card-shadow)"/>
    <!-- Small Tag Badge -->
    <rect x="710" y="946" width="180" height="24" rx="6" fill="#10b981"/>
    <text x="800" y="962" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="11" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="1">CORE PROPOSED MODEL</text>
    <text x="800" y="1005" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="20" font-weight="bold" fill="#0f172a" text-anchor="middle">Design GV-YOLO26 Architecture</text>
    <text x="800" y="1038" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="16" font-weight="600" fill="#047857" text-anchor="middle">GhostConv  •  VoVGSCSPC  •  High-Res P2 Level  •  LGDH Heads</text>
  </g>

  <!-- Arrow: Step 5 -> Step 6 -->
  <line x1="800" y1="1065" x2="800" y2="1205" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 6: Train each design stage -->
  <g id="step6">
    <rect x="360" y="1210" width="880" height="106" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="1257" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Train each design stage with NWD loss and augmentation</text>
    <text x="800" y="1285" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="15" font-weight="normal" fill="#64748b" text-anchor="middle">(6-stage progressive ablation ladder + weather simulation)</text>
  </g>

  <!-- Arrow: Step 6 -> Step 7 -->
  <line x1="800" y1="1316" x2="800" y2="1370" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 7: Distil from teacher -->
  <g id="step7">
    <rect x="360" y="1375" width="880" height="106" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="1422" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Distil from GV-YOLO26-S teacher (BCKD + CWD)</text>
    <text x="800" y="1450" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="15" font-weight="normal" fill="#64748b" text-anchor="middle">(Feature imitation &amp; response distillation negative-result analysis)</text>
  </g>

  <!-- Arrow: Step 7 -> Step 8 -->
  <line x1="800" y1="1481" x2="800" y2="1660" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 8: Evaluate ablation stages -->
  <g id="step8">
    <rect x="360" y="1665" width="880" height="106" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="1712" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Evaluate ablation stages and compare with baselines</text>
    <text x="800" y="1740" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="15" font-weight="normal" fill="#64748b" text-anchor="middle">(Held-out clean validation split &amp; raw test set benchmarking)</text>
  </g>

  <!-- Arrow: Step 8 -> Step 9 -->
  <line x1="800" y1="1771" x2="800" y2="1825" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- Step 9: Export to TensorRT -->
  <g id="step9">
    <rect x="360" y="1830" width="880" height="106" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.8" filter="url(#card-shadow)"/>
    <text x="800" y="1877" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="18" font-weight="600" fill="#1e293b" text-anchor="middle">Export to TensorRT and run on road images and video</text>
    <text x="800" y="1905" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="15" font-weight="normal" fill="#64748b" text-anchor="middle">(NMS-free FP16/INT8 engine + ByteTrack temporal tracking HUD)</text>
  </g>

  <!-- Arrow: Step 9 -> End -->
  <line x1="800" y1="1936" x2="800" y2="2060" stroke="#475569" stroke-width="2.5" marker-end="url(#arrow)"/>

  <!-- END Terminal Node -->
  <g id="node-end">
    <rect x="670" y="2065" width="260" height="64" rx="32" fill="url(#end-grad)" filter="url(#pill-shadow)"/>
    <text x="800" y="2105" font-family="'DejaVu Sans', 'Liberation Sans', Arial, sans-serif" font-size="20" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="1">End</text>
  </g>

</svg>
'''

with open('project_workflow_diagram.svg', 'w', encoding='utf-8') as f:
    f.write(svg_content)

subprocess.run(['rsvg-convert', '-o', 'project_workflow_diagram.png', 'project_workflow_diagram.svg'])
print("Created project_workflow_diagram.png successfully.")
