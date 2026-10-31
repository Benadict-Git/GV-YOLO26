#!/usr/bin/env python3
"""
Generate a clean, publication-grade Black & White Flowchart for GV-YOLO26
styled cleanly as if drawn in draw.io without any color fill or embedded headers.
"""

import html
import subprocess

def esc(s):
    if not isinstance(s, str):
        return str(s)
    return html.escape(s, quote=False)

def create_bw_flowchart_svg(output_svg_path):
    # Canvas dimensions (compact, no title banner)
    W = 860
    H = 1020
    CX = 430  # Center X
    
    # Pure Black and White style constants
    BORDER_COLOR = "#000000"
    CONN_COLOR = "#000000"

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    
    # Definitions: Arrow markers (black and white)
    svg.append('''
    <defs>
      <!-- Draw.io black arrowhead -->
      <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#000000" />
      </marker>
    </defs>
    
    <style>
      text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
      .capsule-text { font-size: 14px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
      .step-title { font-size: 13px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
      .step-sub { font-size: 11px; font-weight: 400; fill: #262626; dominant-baseline: central; text-anchor: middle; }
      .diamond-text { font-size: 11.5px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
      .conn-label { font-size: 10px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
    </style>
    ''')
    
    # White background
    svg.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF" />')

    # Card dimensions
    BW = 490
    BH = 56
    BX = CX - BW // 2  # 430 - 245 = 185
    
    # 1. Start Capsule
    start_y = 30
    start_h = 38
    start_w = 160
    start_x = CX - start_w // 2
    svg.append(f'''
    <!-- START CAPSULE -->
    <g transform="translate({start_x}, {start_y})">
      <rect x="0" y="0" width="{start_w}" height="{start_h}" rx="{start_h//2}" ry="{start_h//2}" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="2" />
      <text x="{start_w//2}" y="{start_h//2}" class="capsule-text">Start</text>
    </g>
    ''')
    
    # Arrow Start -> Step 1
    arrow1_y1 = start_y + start_h
    arrow1_y2 = 92
    svg.append(f'''
    <line x1="{CX}" y1="{arrow1_y1}" x2="{CX}" y2="{arrow1_y2}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    ''')

    # Flowchart Steps
    steps = [
        {
            "num": "1", "y": 92,
            "title": "1. Dataset Curation & Leakage Filtering",
            "sub": "Curate SBP-YOLO dataset (speed12); isolate 482 clean holdout images"
        },
        {
            "num": "2", "y": 174,
            "title": "2. Baseline Benchmarking & Model Selection",
            "sub": "Train & evaluate 7 stock detectors; select YOLO26n as foundational base"
        },
        {
            "num": "3", "y": 256,
            "title": "3. GV-YOLO26 Architectural Engineering",
            "sub": "Incorporate P2 high-res branch (160×160), GhostConv, VoVGSCSPC & LGDH"
        },
        {
            "num": "4", "y": 338,
            "title": "4. Model Training with Composite NWD Loss",
            "sub": "Train with CIoU + Normalized Wasserstein Distance & weather augmentation"
        },
        {
            "num": "5", "y": 420,
            "title": "5. Teacher-Student Knowledge Distillation",
            "sub": "Investigate BCKD & CWD feature alignment from GV-YOLO26-S teacher"
        },
        {
            "num": "6", "y": 502,
            "title": "6. Progressive Multi-Stage Ablation Evaluation",
            "sub": "Evaluate 6 design stages on clean split; identify optimal S6 checkpoint"
        }
    ]

    for idx, s in enumerate(steps):
        sy = s["y"]
        svg.append(f'''
        <!-- STEP {s['num']} -->
        <g id="step-{s['num']}" transform="translate({BX}, {sy})">
          <rect x="0" y="0" width="{BW}" height="{BH}" rx="6" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1.6" />
          <text x="{BW//2}" y="19" class="step-title">{esc(s['title'])}</text>
          <text x="{BW//2}" y="38" class="step-sub">{esc(s['sub'])}</text>
        </g>
        ''')
        
        # Arrow to next step
        if idx < len(steps) - 1:
            next_y = steps[idx+1]["y"]
            svg.append(f'''
            <line x1="{CX}" y1="{sy + BH}" x2="{CX}" y2="{next_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
            ''')

    # Arrow from Step 6 to Decision Diamond
    arrow_d_y1 = steps[-1]["y"] + BH  # 502 + 56 = 558
    diamond_y = 588
    svg.append(f'''
    <line x1="{CX}" y1="{arrow_d_y1}" x2="{CX}" y2="{diamond_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    ''')

    # DECISION DIAMOND
    DW = 280
    DH = 74
    d_mid_y = diamond_y + DH // 2  # 588 + 37 = 625
    d_bot_y = diamond_y + DH       # 588 + 74 = 662
    
    svg.append(f'''
    <!-- DECISION DIAMOND -->
    <g id="decision-diamond" transform="translate({CX}, {d_mid_y})">
      <polygon points="0,{-DH//2} {DW//2},0 0,{DH//2} {-DW//2},0" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1.8" />
      <text y="-8" class="diamond-text">Meets Real-Time (≤15 ms) &amp;</text>
      <text y="10" class="diamond-text">Target Accuracy (&gt;80% mAP)?</text>
    </g>
    ''')

    # NO Branch (Feedback loop back to Step 3)
    loop_x = 755
    step3_mid_y = steps[2]["y"] + BH // 2  # 256 + 28 = 284
    step3_right_x = BX + BW                # 185 + 490 = 675
    
    label_w = 150
    label_h = 22
    label_x = CX + DW // 2 + 15            # 430 + 140 + 15 = 585
    label_y = d_mid_y - label_h // 2       # 625 - 11 = 614
    
    svg.append(f'''
    <!-- NO FEEDBACK LOOP (Clean clearance from Step 6) -->
    <g id="feedback-loop">
      <!-- Path routing: Diamond right vertex -> loop_x -> Step 3 mid_y -> Step 3 right edge -->
      <path d="M {CX + DW//2} {d_mid_y} L {loop_x} {d_mid_y} L {loop_x} {step3_mid_y} L {step3_right_x} {step3_mid_y}" fill="none" stroke="{BORDER_COLOR}" stroke-width="1.6" stroke-dasharray="5,4" stroke-linejoin="round" marker-end="url(#arrow)" />
      
      <!-- 'No (Refine Architecture)' label cleanly placed on horizontal line -->
      <g transform="translate({label_x}, {label_y})">
        <rect x="0" y="0" width="{label_w}" height="{label_h}" rx="4" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1" />
        <text x="{label_w//2}" y="{label_h//2}" class="conn-label">No (Refine Architecture)</text>
      </g>
    </g>
    ''')

    # YES Branch (Continues to Step 7)
    step7_y = 698
    svg.append(f'''
    <!-- YES CONNECTOR -->
    <line x1="{CX}" y1="{d_bot_y}" x2="{CX}" y2="{step7_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    
    <!-- 'Yes' label pill placed cleanly in the gap between Diamond and Step 7 -->
    <g transform="translate({CX + 14}, {(d_bot_y + step7_y) // 2 - 11})">
      <rect x="0" y="0" width="170" height="22" rx="4" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1" />
      <text x="85" y="11" class="conn-label">Yes (84.97% mAP50, 6.2 GF)</text>
    </g>
    ''')

    # Deployment steps
    deploy_steps = [
        {
            "num": "7", "y": 698,
            "title": "7. Embedded TensorRT Engine Compilation",
            "sub": "Export FP16/INT8 NMS-free fused engine for NVIDIA Jetson Xavier (5.8 ms)"
        },
        {
            "num": "8", "y": 780,
            "title": "8. Video Multi-Object Tracking & Confirmation",
            "sub": "ByteTrack temporal tracking; 3-frame confirmation filter eliminates false alarms"
        },
        {
            "num": "9", "y": 862,
            "title": "9. Active Vehicle Suspension Preview & HUD Alert",
            "sub": "Transmit preview signal to soften dampers; render driver HUD alert & buzzer"
        }
    ]

    for idx, s in enumerate(deploy_steps):
        sy = s["y"]
        svg.append(f'''
        <!-- STEP {s['num']} -->
        <g id="step-{s['num']}" transform="translate({BX}, {sy})">
          <rect x="0" y="0" width="{BW}" height="{BH}" rx="6" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1.6" />
          <text x="{BW//2}" y="19" class="step-title">{esc(s['title'])}</text>
          <text x="{BW//2}" y="38" class="step-sub">{esc(s['sub'])}</text>
        </g>
        ''')
        
        # Arrow to next deployment step
        if idx < len(deploy_steps) - 1:
            next_y = deploy_steps[idx+1]["y"]
            svg.append(f'''
            <line x1="{CX}" y1="{sy + BH}" x2="{CX}" y2="{next_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
            ''')

    # Arrow from Step 9 to End Capsule
    end_y = 950
    end_h = 38
    end_w = 160
    end_x = CX - end_w // 2
    svg.append(f'''
    <line x1="{CX}" y1="{deploy_steps[-1]['y'] + BH}" x2="{CX}" y2="{end_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    ''')

    # END CAPSULE
    svg.append(f'''
    <!-- END CAPSULE -->
    <g transform="translate({end_x}, {end_y})">
      <rect x="0" y="0" width="{end_w}" height="{end_h}" rx="{end_h//2}" ry="{end_h//2}" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="2" />
      <text x="{end_w//2}" y="{end_h//2}" class="capsule-text">End</text>
    </g>
    ''')

    svg.append('</svg>')
    
    with open(output_svg_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"SVG written to {output_svg_path}")

if __name__ == '__main__':
    svg_path = "/home/benny/Projects/GV-YOLO26/system_workflow_flowchart.svg"
    png_path = "/home/benny/Projects/GV-YOLO26/system_workflow_flowchart.png"
    create_bw_flowchart_svg(svg_path)
    
    # Render with rsvg-convert at high resolution (1720 width)
    cmd = ["rsvg-convert", "-w", "1720", "--format", "png", "-o", png_path, svg_path]
    subprocess.run(cmd, check=True)
    print(f"B&W Flowchart rendered to {png_path}")
    
    # Also update system_workflow_drawio files
    subprocess.run(["cp", png_path, "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.png"], check=True)
    subprocess.run(["cp", svg_path, "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.svg"], check=True)
