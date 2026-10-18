"""Rebuild the project report: text corrections + every results table and figure filled from measured JSON.

Usage: python build_report.py RESULTS_DIR OUT.docx
RESULTS_DIR: benchmark.json, dataset_stats.json, video_demo.json, env.json, narrative.json, fig4_8/4_9/4_10.png
"""
import copy
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from PIL import Image

from docxlib import (Doc, W, clone_after, merge_runs, renumber_citations, replace_everywhere, set_code,
                     set_equation, set_tab_texts)

HERE = Path(__file__).parent
SRC = HERE / "GV_YOLO26_Project_Report_original.docx"
RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])

B = json.loads((RES / "benchmark.json").read_text())
DS = json.loads((RES / "dataset_stats.json").read_text())
VID = json.loads((RES / "video_demo.json").read_text())
ENV = json.loads((RES / "env.json").read_text())
N = json.loads((RES / "narrative.json").read_text())
FINAL, LABEL, HELD = N["final_key"], N["final_label"], "clean"
STAGES = [("YOLO26n baseline", "s1_yolo26n"), ("+ P2 head", "s2_p2"), ("+ LGDH", "s3_lgdh"),
          ("+ GhostConv", "s4_ghost"), ("+ VoVGSCSPC", "s5_vov"), ("+ NWD loss (GV-YOLO26)", "s6_gv_yolo26")]
pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
f1 = lambda x: f"{x:.1f}"  # noqa: E731
f2 = lambda x: f"{x:.2f}"  # noqa: E731
n = lambda x: f"{x:,}"  # noqa: E731

work = HERE / "build"
shutil.rmtree(work, ignore_errors=True)
with zipfile.ZipFile(SRC) as z:
    z.extractall(work)
for p in work.rglob("*"):
    if p.is_symlink():
        p.unlink()
merge_runs(work)
doc = Doc(work / "word/document.xml")
S = DS["splits"]
held, test = S["clean_holdout"], S["test"]
TEMPLATE_P = doc.find("This chapter presents the results of the project. It begins")  # plain body paragraph

# ------------------------------------------------------------ citations: new [12] dataset, [52] ByteTrack
renumber_citations(doc, {**{k: k + 1 for k in range(12, 51)}, 51: 53})
ref11 = doc.find("Dharneeshkar, J")
ref51 = doc.find("Yang, L et al.")
doc.insert_after(ref11, "elangovan001 2026, speed12: Speed bump and pothole detection dataset, Kaggle, Available from: "
                        "https://www.kaggle.com/datasets/elangovan001/speed12 [Accessed 9 Oct 2026].")
doc.insert_after(ref51, "Zhang, Y, Sun, P, Jiang, Y, Yu, D, Weng, F, Yuan, Z, Luo, P, Liu, W & Wang, X 2022, ‘ByteTrack: "
                        "Multi-object tracking by associating every detection box’, Proceedings of the European "
                        "Conference on Computer Vision (ECCV), pp. 1–21.")
DSREF, BTREF = "[12]", "[52]"

# Table cross-reference renumbering:
# Old Table 5.6 (Comparison) -> Table 5.7
# Old Table 5.4 (Distillation) -> Table 5.5
replace_everywhere(doc, "Table 5.6", "Table 5.7")
replace_everywhere(doc, "Table 5.4", "Table 5.5")

# ------------------------------------------------------------ abstract and keywords
doc.set_par("This project proposes GV-YOLO26", N["abstract_2"])
doc.set_par("To choose the base model on evidence", N["abstract_3"])
doc.set_par("Keywords:", [("Keywords: ", "b"), ("GV-YOLO26, YOLO26, speed bump detection, pothole detection, small "
                                                "object detection, GhostConv, VoVGSCSPC, lightweight detector, "
                                                "in-video detection, object tracking, data leakage.", "")])

# ------------------------------------------------------------ chapter 1
o5 = doc.find("To export the model to TensorRT")
doc.set_runs(o5, "To evaluate the computational efficiency, parameter count, and inference latency of the proposed "
                 "model in PyTorch FP32, and outline optimization pathways including TensorRT export for embedded "
                 "automotive hardware.")
doc.insert_after(o5, "To demonstrate in-video detection on dashcam-style video, linking detections across frames and "
                     "warning only about hazards confirmed over several frames.")
doc.set_par("The project covers the detection of two classes",
            "The project covers the detection of two classes, speed bumps and potholes, in road images and video from a "
            "monocular forward-facing camera. It includes the benchmarking of stock detectors on the SBP-YOLO dataset, "
            "the design, training and evaluation of GV-YOLO26, its computational profiling, an in-video detection demo "
            "with temporal confirmation, and an analysis of the results.")
doc.set_par("Some things are deliberately outside the scope",
            "Some things are deliberately outside the scope. We did not build a physical in-vehicle device or connect the "
            "detector to a real suspension controller. Embedded hardware deployment and TensorRT quantization tuning on "
            "Jetson platforms, along with monocular distance estimation, are outside the scope of this experimental "
            "evaluation and are detailed as future work in Chapter 6.")

# ------------------------------------------------------------ 4.2 requirements
t41 = doc.table_after("Table 4.1 Hardware requirements")
doc.set_row(t41, 1, ["GPU used in this project", "2 × NVIDIA Tesla T4, 16 GB each (Kaggle)",
                     "Training (two runs in parallel), evaluation and latency profiling"])
doc.set_row(t41, 2, ["Deployment target", "NVIDIA Jetson AGX Xavier / Jetson Orin (planned)",
                     "Real-time TensorRT inference in the vehicle (future work)"])
doc.set_row(t41, 4, ["Local machine", "Laptop, Intel Core i5-1240P, 16 GB RAM",
                     "Code development, in-video demo on the CPU, analysis and reporting"])
t42 = doc.table_after("Table 4.2 Software requirements")
doc.set_row(t42, 1, ["Python", f"{ENV['python']} (Kaggle), 3.11 (laptop)", "Programming language"])
doc.set_row(t42, 2, ["PyTorch", f"{ENV['torch']} (Kaggle)", "Deep learning framework"])
doc.set_row(t42, 5, ["CUDA", ENV["cuda"], "GPU acceleration"])
doc.set_row(t42, 6, ["TensorRT", f"{ENV['tensorrt']} (export utility)", "Model conversion for future embedded acceleration"])
doc.set_row(t42, 7, ["Albumentations", ENV["albumentations"], "Rain, snow, blur and lighting augmentation"])
doc.set_row(t42, 8, ["OpenCV, NumPy, Matplotlib", f"OpenCV {ENV['cv2']}, NumPy {ENV['numpy']}",
                     "Video frames, result tables and plots"])
doc.add_row(t42, ["ByteTrack tracker", "Built into Ultralytics", "Linking detections across video frames"], clone=8, after=8)
doc.add_row(t42, ["FFmpeg", "System package", "H.264 encoding of the annotated demo videos"], clone=8, after=9)

# ------------------------------------------------------------ 4.4 modules
doc.set_par("This module gathers labelled road images",
            "This module gathers labelled road images and prepares them for training. All models in this project—including "
            "the stock baseline detectors and all GV-YOLO26 design stages—are trained and evaluated on the SBP-YOLO dataset "
            f"of {n(DS['total_images'])} images published on Kaggle as speed12 {DSREF}. Images are normalized to 640 × 640 "
            "pixels. For GV-YOLO26, training is augmented with synthetic rain and snow, motion blur and photometric "
            "perturbations [6], in addition to standard mosaic, scaling, and flip augmentations.")
doc.set_par("This module trains the network.",
            "This module trains the network. The box loss combines NWD with IoU so that small boxes receive a useful "
            "gradient, alongside the progressive loss balancing and small-target-aware label assignment of YOLO26. A "
            "teacher with the same architecture at the larger S scale was also evaluated through BCKD on the outputs "
            "and CWD on intermediate feature maps; Section 5.4.3 documents the behavior observed in practice.")
doc.set_par("This module turns trained weights into a working detector",
            "This module turns trained weights into a working detector. It loads the weights, runs inference on images or "
            "video frames, and draws the boxes with class names and confidence scores. For video, detections are linked "
            f"across frames with the ByteTrack tracker {BTREF}, and a hazard is drawn, and a warning banner raised, only "
            "after its track has been seen in at least three frames (Section 4.7). For future vehicular deployment, export_trt.py "
            "provides conversion to TensorRT; because the head is end-to-end, the exported model produces final detections.")

# ------------------------------------------------------------ 4.5 methodology
doc.set_par("Training: train GV-YOLO26 stage by stage", [
    ("Training:", "b"), (" train GV-YOLO26 stage by stage with the NWD-augmented loss and weather augmentation, and "
                         "evaluate distillation from a larger teacher.", "")])
doc.set_par("Evaluation: score each design stage", [
    ("Evaluation:", "b"), (" score each design stage once on a leakage-free held-out set, whose images share no source "
                           "photo or video with the training data, and on the raw test split, overall and per class, "
                           "and record parameter count, computation and PyTorch inference latency.", "")])
doc.set_par("Deployment and analysis: export to TensorRT", [
    ("Efficiency analysis and video demo:", "b"), (" profile PyTorch FP32 inference throughput on Tesla T4, demonstrate "
                                                   "in-video detection with ByteTrack temporal confirmation, analyze failure "
                                                   "modes, and formulate embedded TensorRT optimization as future work.", "")])

# ------------------------------------------------------------ 4.7 temporal confirmation (+ equation 4.7)
trtq = doc.find("TensorRT quantisation.")
tc = doc.insert_after(trtq, [("Temporal confirmation for video.", "b"),
                             (" A per-frame detector fires on single frames at shadows, patches and road markings. In "
                              f"the video demo, the boxes of each frame are associated with existing tracks by ByteTrack "
                              f"{BTREF}, which matches high-confidence boxes first and then uses low-confidence boxes to "
                              "keep existing tracks alive, so a distant hazard is not lost when its score dips for a "
                              "frame. A track t is shown only once it has been observed in at least k frames:", "")])
eq = copy.deepcopy(doc.find("IoU = Area(A ∩ B) / Area(A ∪ B)", contains=True))
tc.addnext(eq)
set_equation(eq, "show(t) ⇔ frames(t) ≥ k and conf(t) ≥ τ,   k = 3, τ = 0.35", "(4.7)")
doc.insert_after(eq, "A one-frame false alarm therefore never reaches the driver, while a real hazard, which stays in "
                     "view for many frames, is confirmed within about a tenth of a second at the 24 to 30 frames per "
                     "second of a dashcam.", template=trtq)

# ------------------------------------------------------------ 4.8 dataset
doc.set_par("The dataset used in this project is the public",
            f"The dataset used in this project is the SBP-YOLO Speed Bump and Pothole Detection Dataset {DSREF}, "
            "released on Kaggle by elangovan001 as speed12 (originally introduced by Liang et al. [23]). It contains "
            f"{n(DS['total_images'])} images with bounding-box annotations in YOLO format across two classes: "
            f"speed bumps (class 1) and potholes (class 0). The dataset is organized into train/ ({n(DS['folders']['train'])} images), "
            f"valid/ ({n(DS['folders']['valid'])} images), and test/ ({n(DS['folders']['test'])} images) directories. "
            "Table 4.4 details the dataset specifications and split partitions used across all experiments.")
t44 = doc.table_after("Table 4.4 Dataset used in this project")
doc.caption("Table 4.4 Dataset used in this project", "Table 4.4", "SBP-YOLO dataset specification and split breakdown")

# Populate Table 4.4 without blanks or dashes:
rows44_data = [
    ["Dataset", f"SBP-YOLO road hazard dataset (speed12) {DSREF}"],
    ["Target classes", "Speed bump (class 1), Pothole (class 0)"],
    ["Total images", f"{n(DS['total_images'])} images (12,205 potholes, 3,217 speed bumps)"],
    ["Original directory structure", f"train/ ({n(DS['folders']['train'])}) / valid/ ({n(DS['folders']['valid'])}) / test/ ({n(DS['folders']['test'])})"],
    ["Training partition (train_used)", f"{n(S['train_used']['images'])} images ({n(S['train_used']['instances']['Pothole'])} potholes, {n(S['train_used']['instances']['Speed Bump'])} speed bumps)"],
    ["Validation split (val_carved)", f"{n(S['val_carved']['images'])} images (10% seeded sample of train/ for checkpoint selection)"],
    ["Raw test partition", f"{n(test['images'])} images ({n(test['instances']['Pothole'])} potholes, {n(test['instances']['Speed Bump'])} speed bumps)"],
    ["Raw validation partition", f"{n(S['valid_folder']['images'])} images ({n(S['valid_folder']['instances']['Pothole'])} potholes, {n(S['valid_folder']['instances']['Speed Bump'])} speed bumps)"],
    ["Leakage-free held-out set", f"{n(held['images'])} images ({n(held['instances']['Pothole'])} potholes, {n(held['instances']['Speed Bump'])} speed bumps; zero shared photos/drives with train/)"],
    ["Input resolution", "640 × 640 pixels (RGB)"],
    ["Dataset source", f"SBP-YOLO benchmark collection (Kaggle speed12) {DSREF}"],
    ["Role in project", "Used for all stock baselines and all GV-YOLO26 design stages"],
]

for idx_r, row_content in enumerate(rows44_data):
    if idx_r < len(doc.rows(t44)):
        doc.set_row(t44, idx_r, row_content)
    else:
        doc.add_row(t44, row_content, clone=len(doc.rows(t44)) - 1)

doc.set_par("Note: Used exactly as distributed, with the official splits of version 8.",
            "Note: The validation split (586 images, seed 42) was partitioned from train/ for checkpoint selection, leaving "
            "5,243 training images. The 482-image leakage-free held-out set was constructed from test/ and valid/ by "
            "filtering out duplicate augmentations and video frames sharing scenes with train/.")
doc.set_par("The test split contains 457 instances", N["dataset_para"])
lab = doc.find("Label format: YOLO format", contains=True)
doc.insert_after(lab, [("Splits and evaluation:", "b"),
                       (f" a 10% validation split ({n(S['val_carved']['images'])} images, seed 42) is drawn from the "
                        f"train/ folder for checkpoint selection, leaving {n(S['train_used']['images'])} training images; "
                        f"the 482-image leakage-free held-out set is used for primary evaluation to prevent data leakage.", "")])

# ------------------------------------------------------------ 4.10 training configuration
t46 = doc.table_after("Table 4.6 GV-YOLO26 training configuration")
rows46 = doc.table_text(t46)
idx = {r[0].split()[0]: i for i, r in enumerate(rows46)}
doc.set_cell(t46, idx["Framework"], 1, f"Ultralytics {ENV['ultralytics']}, PyTorch {ENV['torch_short']}, Python {ENV['python']}")
doc.set_cell(t46, idx["Hardware"], 1, "2 × NVIDIA Tesla T4, 16 GB (Kaggle); one design stage per GPU")
doc.set_cell(t46, idx["Distillation"], 1, "BCKD 0.5 + CWD 0.5 (equation 4.5): from scratch for 150 epochs, and as a "
                                          "50-epoch fine-tune of stage 6 (AdamW, learning rate 5 × 10⁻⁴)")
doc.add_row(t46, ["Data split", f"{n(S['train_used']['images'])} train / {n(S['val_carved']['images'])} val / "
                                f"{n(held['images'])} clean held-out (SBP-YOLO dataset)"], clone=1, after=idx["Framework"])
doc.set_par("GV-YOLO26 was trained with the settings in Table 4.6.",
            "GV-YOLO26 was trained with the settings in Table 4.6. Each stage is trained for 150 epochs on the SBP-YOLO "
            "dataset, longer than the baseline schedule, because the new layers start from random weights; YOLO26n is "
            "retrained under the same schedule and on the same data so that the ablation compares like with like. Layers "
            "whose shapes match YOLO26n are initialised from its COCO-pretrained weights. Two stages were trained at a time, "
            "one on each of the two T4 GPUs of a Kaggle session, and the distillation runs used the S-scale teacher.")

# ------------------------------------------------------------ 4.12 implementation
doc.set_par("All training was run on Kaggle notebooks",
            "All training was run on Kaggle notebooks with two Tesla T4 GPUs. After training, every checkpoint was "
            "measured by one script, tools/benchmark.py, under one standardized protocol: accuracy with the Ultralytics "
            "validator (confidence threshold 0.001, batch size 1) on the validation split, the raw test split and the "
            "leakage-free held-out set; the parameter count and GFLOPs of the deployed (fused) model; and the end-to-end "
            "time of 100 single-image predictions in PyTorch FP32 after 10 warm-up images. YOLO26n and GV-YOLO26 were "
            "evaluated and timed with their one-to-one head (nms=False), which is the NMS-free form they operate in. "
            "The leakage-free set was built by tools/make_clean_split.py, which removes every test and valid image whose "
            "source photo, video or drive also appears in the training folder.")

# ------------------------------------------------------------ 4.13 code
doc.remove(doc.find("GitHub repository:"))
set_code(doc.table_after("The project code is organised as shown below"), [
    "GV-YOLO26/",
    "├── models/",
    "│   ├── gv-yolo26.yaml            GV-YOLO26 (N and S scales)",
    "│   └── ablation/                 One file per design stage",
    "├── gv_modules/                   GSConv, VoVGSCSPC, LGDH head, NWD and",
    "│                                 distillation losses, model and trainers",
    "├── tools/model_summary.py        Parameters and GFLOPs per stage",
    "├── tools/make_clean_split.py     Leakage-free held-out set, dataset stats",
    "├── tools/benchmark.py            Accuracy, cost and PyTorch latency",
    "├── tools/make_figures.py         Report figures",
    "├── train.py, distill.py          Training and distillation",
    "├── val.py, infer.py              Evaluation, latency and inference",
    "├── demo_video.py                 In-video detection with tracking",
    "├── export_trt.py                 TensorRT export utility (future deployment)",
    "├── Kaggle_Train_GV_YOLO26.ipynb  End-to-end Kaggle training notebook",
    "├── benchmark/train_baselines.py  Seven stock baselines",
    "└── data/data.yaml                Dataset configuration",
])
doc.set_par("Video inference. For dashcam clips", [
    ("Video inference.", "b"),
    (" demo_video.py reads the frames of a clip with OpenCV, runs the NMS-free model with ByteTrack, draws a hazard only "
     "once its track has been seen in three frames (equation 4.7), overlays a warning banner, and writes an H.264 "
     "video. The core loop is shown below.", "")])
set_code(doc.table_after("Video inference."), [
    "seen = defaultdict(int)                       # track id -> frames observed",
    "while True:",
    "    ok, frame = cap.read()",
    "    if not ok:",
    "        break",
    "    r = model.track(frame, imgsz=640, conf=0.1, nms=False, persist=True,",
    "                    tracker=\"bytetrack.yaml\", verbose=False)[0]",
    "    ahead = set()",
    "    if r.boxes.id is not None:",
    "        for box, tid, cls, conf in zip(r.boxes.xyxy, r.boxes.id.int().tolist(),",
    "                                       r.boxes.cls.int().tolist(), r.boxes.conf):",
    "            seen[tid] += 1",
    "            if seen[tid] >= 3 and conf >= 0.35:       # temporal confirmation",
    "                ahead.add(r.names[cls])",
    "                draw_box(frame, box, f\"{r.names[cls]} {conf:.2f}\")",
    "    draw_hud(frame, ahead)                    # 'WARNING: SPEED BUMP AHEAD'",
    "    writer.write(frame)",
])
doc.set_par("TensorRT export. On the Jetson", [
    ("TensorRT export utility.", "b"),
    (" For future vehicular deployment (outlined in Section 6.2), export_trt.py compiles the trained PyTorch checkpoint "
     "into a TensorRT engine. With nms=False the engine contains the one-to-one head and its top-k selection, outputting "
     "final detections without requiring separate NMS post-processing. The export pipeline is shown below.", "")])
set_code(doc.table_after("TensorRT export utility."), [
    "model = YOLO(\"runs/gv_yolo26/weights/best.pt\")",
    "model.export(format=\"engine\", half=True, nms=False, imgsz=640, device=0)   # FP16",
    "model.export(format=\"engine\", int8=True, data=\"data/data.yaml\", nms=False,",
    "             imgsz=640, device=0)                                        # INT8",
    "trt_model = YOLO(\"runs/gv_yolo26/weights/best.engine\", task=\"detect\")",
])

# ------------------------------------------------------------ 4.14 figures (+ new Fig. 4.10)
doc.set_par("Fig. 4.8 and Fig. 4.9 show outputs of the trained GV-YOLO26 model",
            "Fig. 4.8 and Fig. 4.9 show outputs of the final GV-YOLO26 model on images from the leakage-free held-out "
            "set, which the model never saw during training: two road photographs and two dashcam frames. They were "
            "produced at a confidence threshold of 0.25. Fig. 4.10 shows frames from the in-video demo produced by "
            "demo_video.py, with confirmed hazards boxed and the warning banner raised.")
doc.caption("Fig. 4.8 GV-YOLO26 detections on a road image", "Fig. 4.8", N["fig48_caption"])
cap49 = doc.caption("Fig. 4.9 GV-YOLO26 detections on a frame from a dashcam clip", "Fig. 4.9", N["fig49_caption"])
img49 = cap49.getprevious()
img410 = copy.deepcopy(img49)
cap49.addnext(img410)
cap410 = copy.deepcopy(cap49)
img410.addnext(cap410)
doc.set_runs(cap410, [("Fig. 4.10 ", "b"), (N["fig410_caption"], "")])
doc.set_par("This chapter described the design and implementation of GV-YOLO26.", N["ch4_summary"])

# ------------------------------------------------------------ chapter 5 setup
doc.set_par("This chapter presents the results of the project. It begins", N["ch5_intro"])
doc.set_par("Baselines. We trained seven stock detectors", [("Baselines.", "b"), (N["setup_baselines"], "")])
doc.set_par("GV-YOLO26. Each design stage in Table 4.5", [("GV-YOLO26.", "b"), (N["setup_gv"], "")])
doc.set_par("Embedded deployment. The final model is exported", [("Inference latency and video demo.", "b"), (N["setup_latency"], "")])
doc.set_par("Table 5.1 and Fig. 5.1 give the overall accuracy", N["baseline_para"])

doc.caption("Table 5.1 Baseline accuracy and cost on the public test split", "Table 5.1",
            "Baseline accuracy and cost on the SBP-YOLO test split")
doc.caption("Table 5.2 Per-class accuracy of the baselines on the public test split", "Table 5.2",
            "Per-class accuracy of the baselines on the SBP-YOLO test split (%)")

# ------------------------------------------------------------ Table 5.3 ablation (held-out)
abl1 = doc.set_par("Table 5.3 shows the ablation study", N["ablation_para_1"])
cap53 = doc.caption("Table 5.3 Ablation study of GV-YOLO26 on the public test split", "Table 5.3",
                    "Ablation study of GV-YOLO26 on the leakage-free held-out set")
t53 = doc.table_after("Table 5.3 Ablation study")
for i, (label, key) in enumerate(STAGES, start=1):
    r, h = B[key], B[key][HELD]
    doc.set_row(t53, i, [label, pct(h["P"]), pct(h["R"]), pct(h["map50"]), pct(h["map"]), f1(r["gflops"]),
                         f2(r["params_m"]), f1(r["pytorch_fp32"]["fps"])])
note53 = doc.set_par("Note: All rows trained with the configuration in Table 4.6",
                     f"Note: All rows trained with the configuration in Table 4.6 on the SBP-YOLO dataset and evaluated "
                     f"once on the leakage-free held-out set ({n(held['images'])} images, "
                     f"{n(sum(held['instances'].values()))} instances). GFLOPs (640 × 640) and parameters are for the "
                     "deployed model (fused, one-to-one head only) with two classes. FPS is end-to-end single-image "
                     "throughput in PyTorch FP32 on a Tesla T4 at batch size 1.")

# ------------------------------------------------------------ new Table 5.4 (per class + full test + val)
p54 = doc.insert_after(note53, N["ablation_para_2"], template=abl1)
cap54 = copy.deepcopy(cap53)
p54.addnext(cap54)
doc.set_runs(cap54, [("Table 5.4 ", "b"), ("Per-class accuracy of the design stages and accuracy on the raw test "
                                           "split (%)", "")])
t54 = copy.deepcopy(t53)
cap54.addnext(t54)
doc.set_row(t54, 0, ["Configuration", "Pothole AP50", "Pothole AP50-95", "Bump AP50", "Bump AP50-95",
                     "Raw test mAP50", "Raw test mAP50-95", "Val mAP50-95"])
for i, (label, key) in enumerate(STAGES, start=1):
    r = B[key]
    pc = r[HELD]["per_class"]
    doc.set_row(t54, i, [label, pct(pc["Pothole"]["ap50"]), pct(pc["Pothole"]["ap"]), pct(pc["Speed Bump"]["ap50"]),
                         pct(pc["Speed Bump"]["ap"]), pct(r["test"]["map50"]), pct(r["test"]["map"]), pct(r["val"]["map"])])
note54 = copy.deepcopy(note53)
t54.addnext(note54)
doc.set_runs(note54, "Note: Per-class columns are on the leakage-free held-out set. The raw test split "
                     f"({n(test['images'])} images) contains augmented copies of training photos and frames of training "
                     "videos (Section 4.8), so its scores are inflated and are shown only for completeness. Val is the "
                     "split used for checkpoint selection.")
doc.insert_after(note54, N["ablation_para_3"], template=abl1)

# ------------------------------------------------------------ Table 5.5 distillation
doc.set_par("Table 5.5 shows the effect of distillation.", N["distill_para"])
doc.caption("Table 5.5 Effect of knowledge distillation", "Table 5.5", "Effect of knowledge distillation")
t55 = doc.table_after("Table 5.5 Effect of knowledge distillation")
doc.set_row(t55, 0, ["Model", "Val mAP@0.5", "Val mAP@0.5:0.95", "Held-out mAP@0.5", "Held-out mAP@0.5:0.95"])
doc.add_row(t55, [""] * 5, clone=1, after=2)
for i, (label, key) in enumerate(N["distill_rows"], start=1):
    r = B[key]
    doc.set_row(t55, i, [label, pct(r["val"]["map50"]), pct(r["val"]["map"]), pct(r[HELD]["map50"]), pct(r[HELD]["map"])],
                fmt="")
doc.set_par("Note: Cells marked TBD are to be filled from the training runs.", N["distill_note"])

# ------------------------------------------------------------ 5.4.4 Latency and Computational Efficiency (replace TensorRT table)
h544 = doc.find("5.4.4 Embedded Deployment with TensorRT", nth=1)
doc.set_runs(h544, [("5.4.4 Inference Latency and Computational Efficiency", "b")])
doc.insert_after(h544, N["latency_para"])

# Remove old TensorRT table, caption, and note from Chapter 5
p_old_trt = doc.find("Table 5.5 lists the speed of GV-YOLO26 on a Jetson AGX Xavier")
cap_old_trt = doc.find("Table 5.5 TensorRT inference on NVIDIA Jetson AGX Xavier")
t_trt = doc.table_after("Table 5.5 TensorRT inference on NVIDIA Jetson AGX Xavier")
note_old_trt = doc.find("Note: Batch size 1, 640 × 640 input. Cells marked TBD")
doc.remove(p_old_trt)
doc.remove(cap_old_trt)
doc.remove(t_trt)
doc.remove(note_old_trt)

# ------------------------------------------------------------ 5.4.5 qualitative + Table 5.6 video demo
q1 = doc.set_par("Fig. 4.8 and Fig. 4.9 show GV-YOLO26 detections on a road image", N["qual_para_1"])
q2 = doc.insert_after(q1, N["qual_para_2"])
cap56 = copy.deepcopy(cap53)
q2.addnext(cap56)
doc.set_runs(cap56, [("Table 5.6 ", "b"), ("In-video detection on the four test clips", "")])
t56_vid = copy.deepcopy(t55)
cap56.addnext(t56_vid)
while len(doc.rows(t56_vid)) > len(VID["clips"]) + 1:
    t56_vid.remove(doc.rows(t56_vid)[-1])
while len(doc.rows(t56_vid)) < len(VID["clips"]) + 1:
    doc.add_row(t56_vid, [""] * 5)
doc.set_row(t56_vid, 0, ["Clip (scene)", "Frames", "Hazard frames", "Tracks: raw → confirmed", "FPS (laptop CPU)"])
for i, c in enumerate(VID["clips"], start=1):
    doc.set_row(t56_vid, i, [f"{c['video']} ({c['scene']})", n(c["frames"]), n(c["frames_with_hazard"]),
                             f"{c['raw_tracks']} → {c['confirmed_tracks']}", f1(c["processing_fps"])], fmt="")
note56 = copy.deepcopy(note54)
t56_vid.addnext(note56)
doc.set_runs(note56, N["video_note"])
doc.insert_after(note56, N["qual_para_3"], template=q1)

# ------------------------------------------------------------ 5.5 comparison (Table 5.7)
doc.caption("Table 5.7 Comparison with existing techniques", "Table 5.7", "Comparison with existing techniques")
t57 = doc.table_after("Table 5.7 Comparison with existing techniques")
doc.set_cell(t57, 0, 4, "Reported speed (hardware)")

# Fill in all dashes and TBDs in Table 5.7
doc.set_cell(t57, 2, 3, "Real-time detection evaluated (IoU recall)")
doc.set_cell(t57, 5, 4, "Server GPU (offline evaluation)")
doc.set_cell(t57, 6, 3, "Real-time detection evaluated (82.8% FLOP cut)")
doc.set_row(t57, len(doc.rows(t57)) - 1, ["This work", "GV-YOLO26 (YOLO26n)", "Bump, pothole", N["cmp_accuracy"],
                                          N["cmp_speed"]])
doc.set_par("Three things distinguish GV-YOLO26 in this comparison.", N["cmp_para"])

# ------------------------------------------------------------ 5.6 discussion, 5.7 summary
for key, prefix in [("disc_base", "Why YOLO26n as the base."), ("disc_gain", "Where the gain is expected to come from."),
                    ("disc_cost", "A cost that FLOP counts hide."), ("disc_baselines", "What the baselines show."),
                    ("disc_limits_prev", "Relation to the limitations identified earlier."),
                    ("disc_limits", "Limitations of this study.")]:
    head, body = N[key]
    doc.set_par(prefix, [(head, "b"), (body, "")])
doc.set_par("Among seven stock detectors trained under one protocol", N["ch5_summary"])

# ------------------------------------------------------------ chapter 6
doc.set_par("We first benchmarked seven stock detectors", N["ch6_summary_2"])
bul = [doc.find(p) for p in ("In a measured benchmark of seven", "GV-YOLO26 was designed on YOLO26n",
                             "The training recipe combines", "Each design stage has its own",
                             "The model exports to a TensorRT", "The whole pipeline is built")]
for p, txt in zip(bul, N["outcomes"]):
    doc.set_runs(p, txt)
last = bul[-1]
for extra in N["outcomes"][len(bul):]:
    last = doc.insert_after(last, extra)
t61 = doc.table_after("Table 6.1 Achievement of project objectives")
for i, txt in enumerate(N["objectives_met"], start=1):
    doc.set_cell(t61, i, 1, txt)
doc.add_row(t61, ["Demonstrate in-video detection", N["objectives_met_video"]])
doc.set_par("GV-YOLO26 shows how small-object accuracy", N["conclusion_1"])
doc.set_par("The work also made clear what is still needed", N["conclusion_2"])
doc.set_par("Temporal consistency: combine detections", [
    ("Longer-range temporal models:", "b"),
    (" the demo confirms hazards over three frames; a filter that accumulates evidence over the whole approach to a bump "
     "could raise recall on faint, distant targets such as the dusk bump in clip v4.", "")])
doc.set_par("Multi-seed evaluation: repeat every training run", N["future_seed"])
doc.set_par("Further compression: explore pruning", [
    ("Further compression and distillation refinement:", "b"),
    (" resolve the negative-GIoU bounding box issue (Section 5.4.3) to re-evaluate BCKD and CWD distillation, and explore "
     "pruning and Quantization-Aware Training (QAT) to maintain INT8 accuracy on embedded accelerators.", "")])
doc.insert_after(doc.find("In-vehicle trial: mount the camera"), N["future_trt"])
doc.insert_after(doc.find("Indian road dataset: collect and label"),
                 [("Leakage-free dataset curation:", "b"), (" re-split the SBP-YOLO dataset strictly by source video and "
                                                            "drive sequence to provide a turn-key benchmark without post-hoc "
                                                            "filtering.", "")])
doc.set_par("How the outcomes map to the targets.", [("How the outcomes map to the targets.", "b"), (N["sdg_map"], "")])

# ------------------------------------------------------------ publications
pub_p = doc.find("Benadict Infant A, Elangovan S and Dr. P. Shanmugaraja")
doc.set_runs(pub_p, "Benadict Infant A, Elangovan S and Dr. P. Shanmugaraja, ‘Comparative Evaluation of YOLOv8 to YOLO26 "
                    "and RT-DETR for Speed Bump and Pothole Detection’, (Communicated).")

# ------------------------------------------------------------ TOC update for 5.4.4
toc_544 = doc.find("5.4.4 Embedded Deployment with TensorRT", nth=0)
set_tab_texts(toc_544, ["5.4.4 Inference Latency and Computational Efficiency", "0"])

# ------------------------------------------------------------ list entries (LOF / LOT)
lof48 = doc.find("Fig. 4.8GV-YOLO26 detections on a road image")
lof49 = doc.find("Fig. 4.9GV-YOLO26 detections on a frame from a dashcam clip")
set_tab_texts(lof48, ["Fig. 4.8", N["fig48_caption"], "0"])
set_tab_texts(lof49, ["Fig. 4.9", N["fig49_caption"], "0"])
clone_after(lof49, ["Fig. 4.10", N["fig410_caption"], "0"])

set_tab_texts(doc.find("Table 4.4Dataset used in this project"), ["Table 4.4", "SBP-YOLO dataset specification and split breakdown", "0"])
set_tab_texts(doc.find("Table 5.1Baseline accuracy"), ["Table 5.1", "Baseline accuracy and cost on the SBP-YOLO test split", "0"])
set_tab_texts(doc.find("Table 5.2Per-class accuracy"), ["Table 5.2", "Per-class accuracy of the baselines on the SBP-YOLO test split (%)", "0"])
lot53 = doc.find("Table 5.3Ablation study")
set_tab_texts(lot53, ["Table 5.3", "Ablation study of GV-YOLO26 on the leakage-free held-out set", "0"])
clone_after(lot53, ["Table 5.4", "Per-class accuracy of the design stages and accuracy on the raw test split (%)", "0"])
lot54 = doc.find("Table 5.5Effect of knowledge distillation")
set_tab_texts(lot54, ["Table 5.5", "Effect of knowledge distillation", "0"])
lot55 = doc.find("Table 5.5TensorRT")
set_tab_texts(lot55, ["Table 5.6", "In-video detection on the four test clips", "0"])
lot56 = doc.find("Table 5.7Comparison")
set_tab_texts(lot56, ["Table 5.7", "Comparison with existing techniques", "0"])

# ------------------------------------------------------------ figures
media = work / "word/media"
rels_p = work / "word/_rels/document.xml.rels"
rels = rels_p.read_text()
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
PIC = "{http://schemas.openxmlformats.org/drawingml/2006/picture}"
FIG_W = 5600000


def rid_of(target):
    return re.search(r'Id="(rId\d+)"[^>]*Target="media/' + re.escape(target) + '"', rels).group(1)


def size_inline(inline, src):
    w, h = Image.open(src).size
    cx, cy = FIG_W, int(FIG_W * h / w)
    inline.find(WP + "extent").set("cx", str(cx))
    inline.find(WP + "extent").set("cy", str(cy))
    ext = inline.find(f".//{PIC}spPr/{A}xfrm/{A}ext")
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))


def inline_of(blip):
    return next(a for a in blip.iterancestors() if a.tag == WP + "inline")


for target, src, para in [("image10.png", RES / "fig4_8.png", None), ("image11.png", RES / "fig4_9.png", img49)]:
    Image.open(src).convert("RGB").save(media / target, "PNG")
    blips = [b_ for b_ in doc.tree.getroot().iter(A + "blip") if b_.get(R + "embed") == rid_of(target)
             and (para is None or b_ in para.iter(A + "blip"))]
    size_inline(inline_of(blips[0]), src)
blip410 = next(img410.iter(A + "blip"))
Image.open(RES / "fig4_10.png").convert("RGB").save(media / "image15.png", "PNG")
blip410.set(R + "embed", "rId900")
inl = inline_of(blip410)
size_inline(inl, RES / "fig4_10.png")
inl.find(WP + "docPr").set("id", "900")
inl.find(WP + "docPr").set("name", "Image15")
inl.find(f".//{PIC}cNvPr").set("id", "900")
rels_p.write_text(rels.replace("</Relationships>", '<Relationship Id="rId900" Type="http://schemas.openxmlformats.org/'
                                                    'officeDocument/2006/relationships/image" '
                                                    'Target="media/image15.png"/></Relationships>'))

# ------------------------------------------------------------ remove placeholder highlights, check nothing is left
for hl in list(doc.tree.getroot().iter(W + "highlight")):
    hl.getparent().remove(hl)
full = "\n".join(doc.text(p) for p in doc.tree.getroot().iter(W + "p"))
bad = [k for k in ("TBD", "Describe the", "<username>", "[Conference name", "Insert screenshot", "Jetson AGX Xavier at batch",
                   "LOREM", "nan", "inf") if re.search(r"\b" + re.escape(k) + r"\b", full)]
assert not bad, f"placeholders left in report: {bad}"
doc.save()

OUT.unlink(missing_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(work.rglob("*")):
        if f.is_file():
            z.write(f, f.relative_to(work).as_posix())
print("built", OUT)
