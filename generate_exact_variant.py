import subprocess

svg_exact = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="1750" viewBox="0 0 1100 1750">
  <defs>
    <filter id="soft-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#0f172a" flood-opacity="0.06"/>
    </filter>
    <filter id="pill-shadow" x="-5%" y="-5%" width="110%" height="115%" filterUnits="userSpaceOnUse">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#1d4ed8" flood-opacity="0.2"/>
    </filter>

    <linearGradient id="blue-pill" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#2563eb"/>
      <stop offset="100%" stop-color="#1d4ed8"/>
    </linearGradient>

    <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">
      <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#334155"/>
    </marker>
  </defs>

  <rect width="100%" height="100%" fill="#ffffff"/>

  <!-- Start -->
  <rect x="430" y="35" width="240" height="54" rx="27" fill="url(#blue-pill)" filter="url(#pill-shadow)"/>
  <text x="550" y="70" font-family="'DejaVu Sans', Arial, sans-serif" font-size="19" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="1">Start</text>

  <line x1="550" y1="89" x2="550" y2="135" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 1 -->
  <rect x="120" y="140" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="177" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Review literature on bump and</text>
  <text x="550" y="205" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">pothole detection</text>

  <line x1="550" y1="224" x2="550" y2="270" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 2 -->
  <rect x="120" y="275" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="312" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Train seven stock baselines on</text>
  <text x="550" y="340" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">public dataset (fixed protocol)</text>

  <line x1="550" y1="359" x2="550" y2="405" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 3 -->
  <rect x="120" y="410" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="447" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Evaluate baselines on</text>
  <text x="550" y="475" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">held-out test split</text>

  <line x1="550" y1="494" x2="550" y2="540" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 4 -->
  <rect x="120" y="545" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="582" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Select YOLO26n as base</text>
  <text x="550" y="610" font-family="'DejaVu Sans', Arial, sans-serif" font-size="16" font-weight="normal" fill="#475569" text-anchor="middle">(most accurate, lowest cost)</text>

  <line x1="550" y1="629" x2="550" y2="675" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 5 -->
  <rect x="120" y="680" width="860" height="92" rx="10" fill="#f8fafc" stroke="#2563eb" stroke-width="2.4" filter="url(#soft-shadow)"/>
  <text x="550" y="720" font-family="'DejaVu Sans', Arial, sans-serif" font-size="18" font-weight="bold" fill="#0f172a" text-anchor="middle">Design GV-YOLO26: GhostConv,</text>
  <text x="550" y="750" font-family="'DejaVu Sans', Arial, sans-serif" font-size="18" font-weight="bold" fill="#2563eb" text-anchor="middle">VoVGSCSPC, P2 level, LGDH</text>

  <line x1="550" y1="772" x2="550" y2="818" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 6 -->
  <rect x="120" y="823" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="860" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Train each design stage with</text>
  <text x="550" y="888" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">NWD loss and augmentation</text>

  <line x1="550" y1="907" x2="550" y2="953" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 7 -->
  <rect x="120" y="958" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="995" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Distil from GV-YOLO26-S</text>
  <text x="550" y="1023" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">teacher (BCKD + CWD)</text>

  <line x1="550" y1="1042" x2="550" y2="1088" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 8 -->
  <rect x="120" y="1093" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="1130" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Evaluate ablation stages</text>
  <text x="550" y="1158" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">and compare with baselines</text>

  <line x1="550" y1="1177" x2="550" y2="1223" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- Box 9 -->
  <rect x="120" y="1228" width="860" height="84" rx="10" fill="#ffffff" stroke="#334155" stroke-width="2" filter="url(#soft-shadow)"/>
  <text x="550" y="1265" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">Export to TensorRT and run</text>
  <text x="550" y="1293" font-family="'DejaVu Sans', Arial, sans-serif" font-size="17" font-weight="500" fill="#0f172a" text-anchor="middle">on road images and video</text>

  <line x1="550" y1="1312" x2="550" y2="1358" stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>

  <!-- End -->
  <rect x="430" y="1363" width="240" height="54" rx="27" fill="#1e293b" filter="url(#soft-shadow)"/>
  <text x="550" y="1398" font-family="'DejaVu Sans', Arial, sans-serif" font-size="19" font-weight="bold" fill="#ffffff" text-anchor="middle" letter-spacing="1">End</text>

</svg>
'''

with open('workflow_diagram_exact.svg', 'w', encoding='utf-8') as f:
    f.write(svg_exact)

subprocess.run(['rsvg-convert', '-o', 'workflow_diagram_exact.png', 'workflow_diagram_exact.svg'])
print("Generated workflow_diagram_exact.png successfully.")
