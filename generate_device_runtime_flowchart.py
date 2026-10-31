#!/usr/bin/env python3
"""
Generate a clean, publication-grade Black & White Device Runtime Flowchart for GV-YOLO26.
Refactored for maximum clarity and understanding:
- Clear, intuitive terminology (no confusing jargon like "ByteTrack association")
- Consolidated action steps (streamlined to 6 core steps, eliminating redundant boxes)
- Pure black & white draw.io styling without color fills or embedded banners
"""

import html
import subprocess

def esc(s):
    if not isinstance(s, str):
        return str(s)
    return html.escape(s, quote=False)

def create_device_runtime_flowchart_svg(output_svg_path):
    # Canvas dimensions
    W = 820
    H = 760
    CX = 410  # Center X
    
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
      .step-title { font-size: 13.5px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
      .step-sub { font-size: 11px; font-weight: 400; fill: #262626; dominant-baseline: central; text-anchor: middle; }
      .diamond-title { font-size: 12.5px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
      .diamond-sub { font-size: 10.5px; font-weight: 500; dominant-baseline: central; text-anchor: middle; fill: #333333; }
      .conn-label { font-size: 10px; font-weight: 700; dominant-baseline: central; text-anchor: middle; fill: #000000; }
    </style>
    ''')
    
    # White background
    svg.append(f'<rect width="{W}" height="{H}" fill="#FFFFFF" />')

    # Card dimensions
    BW = 480
    BH = 56
    BX = CX - BW // 2  # 410 - 240 = 170
    
    # 1. Start Capsule
    start_y = 26
    start_h = 36
    start_w = 150
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
    arrow1_y2 = 86
    svg.append(f'''
    <line x1="{CX}" y1="{arrow1_y1}" x2="{CX}" y2="{arrow1_y2}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    ''')

    # Pre-decision Steps (1 to 4) - Refactored for clear understanding
    steps_before_decision = [
        {
            "num": "1", "y": 86,
            "title": "1. Video Frame Acquisition",
            "sub": "Capture live road video stream from front-facing vehicle camera"
        },
        {
            "num": "2", "y": 166,
            "title": "2. Image Preprocessing",
            "sub": "Resize frames to 640×640 and enhance road surface visibility"
        },
        {
            "num": "3", "y": 246,
            "title": "3. Road Hazard Detection (GV-YOLO26)",
            "sub": "Detect speed bumps and potholes in real time with bounding boxes"
        },
        {
            "num": "4", "y": 326,
            "title": "4. Hazard Tracking & Temporal Verification",
            "sub": "Track detected objects across consecutive frames to filter out false alarms"
        }
    ]

    for idx, s in enumerate(steps_before_decision):
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
        if idx < len(steps_before_decision) - 1:
            next_y = steps_before_decision[idx+1]["y"]
            svg.append(f'''
            <line x1="{CX}" y1="{sy + BH}" x2="{CX}" y2="{next_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
            ''')

    # Arrow from Step 4 to Decision Diamond
    arrow_d_y1 = steps_before_decision[-1]["y"] + BH  # 326 + 56 = 382
    diamond_y = 408
    svg.append(f'''
    <line x1="{CX}" y1="{arrow_d_y1}" x2="{CX}" y2="{diamond_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    ''')

    # DECISION DIAMOND: Simple verification check
    DW = 280
    DH = 74
    d_mid_y = diamond_y + DH // 2  # 408 + 37 = 445
    d_bot_y = diamond_y + DH       # 408 + 74 = 482
    
    svg.append(f'''
    <!-- DECISION DIAMOND: Hazard Confirmation -->
    <g id="decision-diamond" transform="translate({CX}, {d_mid_y})">
      <polygon points="0,{-DH//2} {DW//2},0 0,{DH//2} {-DW//2},0" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1.8" />
      <text y="-8" class="diamond-title">Hazard Confirmed?</text>
      <text y="10" class="diamond-sub">(Seen in ≥ 3 Frames)</text>
    </g>
    ''')

    # NO Branch (Ignore false alarm / shadow -> continue road monitoring)
    loop_x = 720
    step1_mid_y = steps_before_decision[0]["y"] + BH // 2  # 86 + 28 = 114
    step1_right_x = BX + BW                               # 170 + 480 = 650
    
    label_w = 160
    label_h = 22
    label_x = CX + DW // 2 + 12                           # 410 + 140 + 12 = 562
    label_y = d_mid_y - label_h // 2                      # 445 - 11 = 434
    
    svg.append(f'''
    <!-- NO FEEDBACK LOOP: Suppress False Alarms -->
    <g id="feedback-loop">
      <!-- Path routing: Diamond right vertex -> loop_x -> Step 1 mid_y -> Step 1 right edge -->
      <path d="M {CX + DW//2} {d_mid_y} L {loop_x} {d_mid_y} L {loop_x} {step1_mid_y} L {step1_right_x} {step1_mid_y}" fill="none" stroke="{BORDER_COLOR}" stroke-width="1.6" stroke-dasharray="5,4" stroke-linejoin="round" marker-end="url(#arrow)" />
      
      <!-- 'No' label cleanly placed on horizontal line -->
      <g transform="translate({label_x}, {label_y})">
        <rect x="0" y="0" width="{label_w}" height="{label_h}" rx="4" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1" />
        <text x="{label_w//2}" y="{label_h//2}" class="conn-label">No (Ignore False Alarm)</text>
      </g>
    </g>
    ''')

    # YES Branch (Continues downward to Distance & Severity Estimation)
    step5_y = 518
    svg.append(f'''
    <!-- YES CONNECTOR -->
    <line x1="{CX}" y1="{d_bot_y}" x2="{CX}" y2="{step5_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
    
    <!-- 'Yes' label pill placed cleanly in the gap between Diamond and Step 5 -->
    <g transform="translate({CX + 14}, {(d_bot_y + step5_y) // 2 - 11})">
      <rect x="0" y="0" width="150" height="22" rx="4" fill="#FFFFFF" stroke="{BORDER_COLOR}" stroke-width="1" />
      <text x="75" y="11" class="conn-label">Yes (Hazard Verified)</text>
    </g>
    ''')

    # Post-confirmation Steps (Consolidated into 2 clear, punchy actions)
    steps_after_decision = [
        {
            "num": "5", "y": 518,
            "title": "5. Distance & Time-to-Impact Estimation",
            "sub": "Calculate longitudinal distance to hazard and estimated vehicle arrival time"
        },
        {
            "num": "6", "y": 598,
            "title": "6. Driver Warning & Suspension Control",
            "sub": "Display visual HUD alert, sound buzzer, and adjust suspension damping"
        }
    ]

    for idx, s in enumerate(steps_after_decision):
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
        if idx < len(steps_after_decision) - 1:
            next_y = steps_after_decision[idx+1]["y"]
            svg.append(f'''
            <line x1="{CX}" y1="{sy + BH}" x2="{CX}" y2="{next_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
            ''')

    # Arrow from Step 6 to End Capsule
    end_y = 684
    end_h = 36
    end_w = 150
    end_x = CX - end_w // 2
    svg.append(f'''
    <line x1="{CX}" y1="{steps_after_decision[-1]['y'] + BH}" x2="{CX}" y2="{end_y}" stroke="{CONN_COLOR}" stroke-width="1.8" marker-end="url(#arrow)" />
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
    create_device_runtime_flowchart_svg(svg_path)
    
    # Render with rsvg-convert at high resolution (1720 width)
    cmd = ["rsvg-convert", "-w", "1720", "--format", "png", "-o", png_path, svg_path]
    subprocess.run(cmd, check=True)
    print(f"Device Runtime Flowchart rendered to {png_path}")
    
    # Also update system_workflow_drawio files
    subprocess.run(["cp", png_path, "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.png"], check=True)
    subprocess.run(["cp", svg_path, "/home/benny/Projects/GV-YOLO26/system_workflow_drawio.svg"], check=True)
