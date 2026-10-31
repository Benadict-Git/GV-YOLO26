#!/usr/bin/env python3
"""Fetch trained weights from Kaggle when training completes and run video test suite.

Monitors the Kaggle kernel until completion, downloads gv_yolo26_x_best.pt and results.json,
and executes GV-YOLO26/demo_video.py across all clips in video/ (v1.mp4 to v4.mp4).
"""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def log(msg: str):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def check_kernel_status(kernel_id: str) -> str:
    """Run `kaggle kernels status` and return the raw status string."""
    cmd = ["kaggle", "kernels", "status", kernel_id]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Warning: kaggle CLI error checking status: {res.stderr.strip()}")
        return "UNKNOWN"
    return res.stdout.strip()


def poll_until_complete(kernel_id: str, interval: int = 60, max_hours: float = 12.0) -> bool:
    """Poll kernel status until complete or failed."""
    start_time = time.time()
    max_seconds = max_hours * 3600
    log(f"Monitoring Kaggle kernel: {kernel_id}")
    log(f"Checking every {interval}s (max wait: {max_hours}h)...")

    while True:
        elapsed_sec = int(time.time() - start_time)
        elapsed_str = f"{elapsed_sec // 3600}h {(elapsed_sec % 3600) // 60}m {elapsed_sec % 60}s"
        status_line = check_kernel_status(kernel_id)
        log(f"Elapsed: {elapsed_str} | {status_line}")

        status_lower = status_line.lower()

        if "complete" in status_lower:
            log(f"Kernel execution finished successfully in {elapsed_str}!")
            return True

        if any(term in status_lower for term in ("error", "failed", "cancel")):
            log(f"Kernel run terminated abnormally: {status_line}")
            return False

        if elapsed_sec > max_seconds:
            log(f"Timeout reached ({max_hours} hours). Stopping poll.")
            return False

        time.sleep(interval)


def download_artifacts(kernel_id: str, dest_dir: Path) -> Path:
    """Download artifacts from the completed Kaggle kernel."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading outputs to {dest_dir}...")
    cmd = ["kaggle", "kernels", "output", kernel_id, "-p", str(dest_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log(f"Error downloading outputs: {res.stderr.strip()}")
        raise RuntimeError(f"kaggle kernels output failed: {res.stderr}")

    log("Download complete. Files received:")
    for p in dest_dir.rglob("*"):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            log(f"  - {p.relative_to(dest_dir)} ({size_mb:.2f} MB)")

    # Display results.json if available
    results_json = dest_dir / "results.json"
    if results_json.exists():
        try:
            metrics = json.loads(results_json.read_text())
            log("\n=== EXPERIMENTAL METRICS SUMMARY ===")
            print(json.dumps(metrics, indent=2))
            log("====================================\n")
        except Exception as e:
            log(f"Notice: could not parse results.json: {e}")

    # Locate best weights file
    candidates = list(dest_dir.rglob("*best*.pt")) + list(dest_dir.rglob("*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No .pt weights found in {dest_dir}")

    # Prioritize gv_yolo26_x_best.pt, then best.pt
    best_weight = None
    for cand in candidates:
        if "gv_yolo26_x" in cand.name:
            best_weight = cand
            break
    if not best_weight:
        best_weight = candidates[0]

    log(f"Selected primary weights: {best_weight}")
    return best_weight


def run_video_tests(weights_path: Path, video_dir: Path, output_dir: Path, conf: float, confirm: int, ego_speed: float):
    """Run GV-YOLO26/demo_video.py on all video clips."""
    log(f"\n=== RUNNING VIDEO TEST SUITE ON {video_dir} ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_script = REPO_ROOT / "GV-YOLO26" / "demo_video.py"
    if not demo_script.exists():
        raise FileNotFoundError(f"demo_video.py not found at {demo_script}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "GV-YOLO26") + ":" + env.get("PYTHONPATH", "")

    cmd = [
        sys.executable,
        str(demo_script),
        "--weights", str(weights_path),
        "--source", str(video_dir),
        "--out", str(output_dir),
        "--conf", str(conf),
        "--confirm", str(confirm),
        "--ego-speed", str(ego_speed),
    ]

    log(f"Executing: {' '.join(cmd)}")
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        log(f"Warning: demo_video.py exited with code {res.returncode}")
    else:
        log("Video test suite executed successfully.")

    # Read and print summary
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            log("\n=== VIDEO DETECTION TEST RESULTS ===")
            print(f"{'Video':<12} {'Frames':<8} {'Haz.Frames':<12} {'Tracks':<8} {'Speed Bumps':<14} {'Potholes':<10} {'FPS':<6}")
            print("-" * 75)
            for item in summary:
                print(
                    f"{item.get('video', ''):<12} "
                    f"{item.get('frames', 0):<8} "
                    f"{item.get('frames_with_hazard', 0):<12} "
                    f"{item.get('confirmed_tracks', 0):<8} "
                    f"{item.get('unique_speed_bumps', 0):<14} "
                    f"{item.get('unique_potholes', 0):<10} "
                    f"{item.get('processing_fps', 0):<6}"
                )
            print("-" * 75)
            log(f"Annotated video outputs saved to: {output_dir}")
        except Exception as e:
            log(f"Could not render summary table: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Kaggle training output and run video evaluations.")
    parser.add_argument("--kernel", default="benadictinfanta/gv-yolo26-x-train", help="Kaggle kernel identifier")
    parser.add_argument("--poll-interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--dest", default="GV-YOLO26/weights_kaggle/gv-yolo26-x", help="Download destination folder")
    parser.add_argument("--target-weights", default="GV-YOLO26/weights/gv_yolo26_x_best.pt", help="Path to copy primary weights")
    parser.add_argument("--video-dir", default="video", help="Directory containing test videos")
    parser.add_argument("--video-out", default="results/video_gvx", help="Directory for annotated video outputs")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--confirm", type=int, default=3, help="Confirmation frame threshold")
    parser.add_argument("--ego-speed", type=float, default=60.0, help="Vehicle speed in km/h for TTI preview")
    parser.add_argument("--skip-wait", action="store_true", help="Skip polling and proceed directly to fetch/test")
    parser.add_argument("--test-only", default=None, help="Skip fetch and test an existing weights file directly")
    args = parser.parse_args()

    dest_dir = Path(args.dest).resolve()
    video_dir = Path(args.video_dir).resolve()
    video_out = Path(args.video_out).resolve()
    target_weights = Path(args.target_weights).resolve()

    if args.test_only:
        weights_file = Path(args.test_only).resolve()
        if not weights_file.exists():
            log(f"Error: provided --test-only weights file does not exist: {weights_file}")
            sys.exit(1)
        run_video_tests(weights_file, video_dir, video_out, args.conf, args.confirm, args.ego_speed)
        return

    # Step 1: Wait for completion
    if not args.skip_wait:
        success = poll_until_complete(args.kernel, interval=args.poll_interval)
        if not success:
            log("Kaggle training did not complete successfully. Exiting.")
            sys.exit(1)

    # Step 2: Download artifacts
    best_weights = download_artifacts(args.kernel, dest_dir)

    # Step 3: Copy to standard weights directory
    target_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(best_weights), str(target_weights))
    log(f"Promoted {best_weights.name} to {target_weights}")

    # Step 4: Run video tests
    run_video_tests(target_weights, video_dir, video_out, args.conf, args.confirm, args.ego_speed)


if __name__ == "__main__":
    main()
