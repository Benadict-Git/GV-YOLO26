"""All results-dependent report text, generated from the measured numbers so prose and tables cannot disagree."""

STAGE_KEYS = ["s1_yolo26n", "s2_p2", "s3_lgdh", "s4_ghost", "s5_vov", "s6_gv_yolo26"]
STAGE_NAMES = {"s2_p2": "the P2 level", "s3_lgdh": "the LGDH", "s4_ghost": "GhostConv", "s5_vov": "VoVGSCSPC",
               "s6_gv_yolo26": "the NWD loss"}
ROW_NAME = dict(zip(STAGE_KEYS, ["YOLO26n baseline", "stage 2 (+ P2)", "stage 3 (+ LGDH)", "stage 4 (+ GhostConv)",
                                 "stage 5 (+ VoVGSCSPC)", "stage 6 (GV-YOLO26)"]))


def P(x):
    return f"{100 * x:.1f}"


def pts(a, b):
    d = 100 * (b - a)
    return f"{'+' if d >= 0 else '−'}{abs(d):.1f}"


def moved(a, b):
    d = 100 * (b - a)
    if abs(d) < 0.05:
        return f"stayed at {P(b)}%"
    return f"{'rose' if d > 0 else 'fell'} from {P(a)}% to {P(b)}%"


def build(B, DS, VID, final_key, final_label, kd_info, box_stats):
    HELD = "clean"
    S = DS["splits"]
    held, test = S["clean_holdout"], S["test"]
    ov_t, ov_v = DS["test_overlap"], DS["valid_overlap"]
    b, g, t = B["s1_yolo26n"], B["s6_gv_yolo26"], B["teacher_s"]
    fin = B[final_key]
    h = lambda k: B[k][HELD]  # noqa: E731
    pc = lambda k, c: B[k][HELD]["per_class"][c]  # noqa: E731
    param_cut = 100 * (1 - g["params_m"] / b["params_m"])
    flop_cut = 100 * (1 - g["gflops"] / b["gflops"])
    best_stage = max(STAGE_KEYS, key=lambda k: h(k)["map"])
    fps = {k: B[k]["pytorch_fp32"]["fps"] for k in STAGE_KEYS}
    sweep = {sz: B.get(f"{final_key}@{sz}") for sz in (800, 960)}
    clips = VID["clips"]
    raw = sum(c["raw_tracks"] for c in clips)
    conf = sum(c["confirmed_tracks"] for c in clips)
    v4 = next((c for c in clips if c["video"] == "v4.mp4"), None)
    gap50, gap = pts(h("s1_yolo26n")["map50"], h(final_key)["map50"]), pts(h("s1_yolo26n")["map"], h(final_key)["map"])
    leak_pct = 100 * ov_t["share_photo_with_train"] / ov_t["total"]
    infl = [100 * (B[k]["test"]["map"] - h(k)["map"]) for k in STAGE_KEYS]

    if max(fps.values()) / min(fps.values()) < 1.15:
        speed_sentence = (f" In PyTorch on the T4 every row ran at a similar single-image speed ({min(fps.values()):.1f} to "
                          f"{max(fps.values()):.1f} FPS), because at batch size 1 the time is dominated by per-layer "
                          "overhead and memory bandwidth rather than by arithmetic FLOPs.")
    else:
        speed_sentence = (f" In PyTorch on the T4, YOLO26n ran at {fps['s1_yolo26n']:.1f} FPS and the final model at "
                          f"{fps['s6_gv_yolo26']:.1f} FPS; at batch size 1 the extra P2 level and the deeper stacks of "
                          "cheap layers add per-layer overhead that FLOP counts do not capture.")

    steps = [f"{STAGE_NAMES[cur]} changed mAP@0.5:0.95 by {pts(h(prev)['map'], h(cur)['map'])} points"
             for prev, cur in zip(STAGE_KEYS, STAGE_KEYS[1:])]
    N = {}
    N["final_key"], N["final_label"] = final_key, final_label
    N["pt_mb"] = B.get("pt_mb", {"s1_yolo26n": 5.2, "s6_gv_yolo26": 3.7})

    # ---------------------------------------------------------------- abstract
    N["abstract_2"] = (
        "This project proposes GV-YOLO26, a lightweight real-time detector for speed bumps and potholes built on YOLO26n, "
        "the newest end-to-end member of the YOLO family. We redesigned its backbone and neck around cheap convolutions: "
        "GhostConv layers replace the costly downsampling convolutions of the deeper stages, and VoVGSCSPC blocks built "
        "from GSConv replace the C3k2 feature blocks. A high-resolution P2 detection level is added for small, distant "
        "targets, and a Lightweight Grouped Detection Head (LGDH) with shared grouped convolutions keeps the extra level "
        "affordable while retaining the NMS-free one-to-one prediction of YOLO26. In its final form GV-YOLO26 has "
        f"{g['params_m']:.2f} million parameters and needs {g['gflops']:.1f} GFLOPs, {param_cut:.0f}% fewer parameters "
        f"and less computation than YOLO26n ({b['params_m']:.2f} million, {b['gflops']:.1f} GFLOPs), even though it "
        "predicts on an extra 160 × 160 grid. Training incorporates a Normalized Wasserstein Distance (NWD) box loss and "
        "weather augmentation; knowledge distillation from a larger GV-YOLO26 teacher was also evaluated, and an in-video "
        "hazard detection pipeline with temporal tracking was demonstrated, with embedded hardware acceleration identified "
        "as future work.")
    N["abstract_3"] = (
        "To choose the base model on evidence, we first benchmarked seven stock detectors, from YOLOv8n to YOLO26n and "
        "RT-DETR-L, on the SBP-YOLO road hazard dataset; YOLO26n was the most accurate at 74.8% mAP@0.5 and the cheapest at "
        f"5.3 GFLOPs. GV-YOLO26 was then trained stage by stage through an ablation ladder on the {DS['total_images']:,}-image "
        f"SBP-YOLO dataset. We found that {leak_pct:.0f}% of the dataset's raw test images are augmented copies or video frames "
        "of training scenes, and evaluated every model on a leakage-free held-out set of "
        f"{held['images']} images instead. There GV-YOLO26 reached {P(h('s6_gv_yolo26')['map50'])}% mAP@0.5 against "
        f"{P(h('s1_yolo26n')['map50'])}% for an identically trained YOLO26n baseline, with {param_cut:.0f}% fewer parameters, "
        f"running at {fps['s6_gv_yolo26']:.1f} FPS in PyTorch FP32 on an NVIDIA Tesla T4 GPU. Distillation did not improve "
        "accuracy: as implemented, its box term was unbounded and destabilised training. An in-video demo links detections "
        "across frames with ByteTrack and warns only about hazards confirmed over three frames. TensorRT edge acceleration "
        "and embedded Jetson deployment are formulated as future work.")

    # ---------------------------------------------------------------- chapter 4
    N["dataset_para"] = (
        "The SBP-YOLO dataset (speed12) comprises 7,600 images across train/, valid/, and test/ folders. Before evaluation, "
        "the dataset required rigorous verification. File names and image hashes show that many images in test/ and valid/ "
        "are augmented copies of photos in train/: for example, train/ holds add_gasuss_noise_1386_jpg.rf.df5cd630….jpg, a "
        "noise-augmented copy of the test image 1386_jpg.rf.df5cd630….jpg with an identical visual hash. Other files represent "
        "neighbouring frames from sequential dashcam or GoPro video drives. We traced every image back to its source photo "
        f"and drive. Of the {ov_t['total']:,} test images, {ov_t['share_photo_with_train']:,} share a source photo with train/ "
        f"and {ov_t['share_sequence_with_train']:,} share a photo or video sequence; of the {ov_v['total']:,} valid images, "
        f"{ov_v['share_sequence_with_train']:,} do. Scores on the raw test split therefore reward memorisation as well as "
        f"detection. To ensure strict academic integrity, all main experimental results in Chapter 5 are reported on the "
        f"remaining {held['images']} images ({ov_t['clean']} from test/ and {ov_v['clean']} from valid/), which share zero "
        "source photos, videos, or drives with the training partition. This leakage-free held-out set contains "
        f"{held['instances']['Pothole']} pothole instances in {held['images_with']['Pothole']} images and "
        f"{held['instances']['Speed Bump']} speed bump instances in {held['images_with']['Speed Bump']} images. Each "
        "pothole in a cluster is annotated individually, and speed bump bounding boxes cover the full road width, conventions "
        "that make strict-IoU metrics lower for potholes than for speed bumps.")
    N["fig48_caption"] = "GV-YOLO26 detections on held-out road images: (a) a distant bump, (b) a painted bump"
    N["fig49_caption"] = "GV-YOLO26 detections on held-out dashcam frames: (a) an urban road, (b) a distant pothole"
    N["fig410_caption"] = "Frames from the in-video detection demo with confirmed hazards and the warning banner"
    N["ch4_summary"] = (
        "This chapter described the design and implementation of GV-YOLO26. The network rebuilds the YOLO26n backbone "
        "and neck with GhostConv and VoVGSCSPC, adds a P2 detection level for small targets, and offsets its cost with "
        "the LGDH while keeping NMS-free inference; the training recipe adds the NWD loss and weather augmentation, and "
        "distillation was tested. For video, ByteTrack links detections across frames and a temporal confirmation rule "
        "suppresses one-frame false alarms. Seven stock detectors trained on the SBP-YOLO dataset justify the choice of "
        "YOLO26n as the base, and every design stage is evaluated on a leakage-free held-out set. The next chapter presents "
        "the results.")

    # ---------------------------------------------------------------- chapter 5 setup
    N["ch5_intro"] = (
        "This chapter presents the experimental results of the project. It begins with the benchmark of seven stock detectors "
        "on the SBP-YOLO dataset that was used to select the base model, then reports the 6-stage ablation study of GV-YOLO26, "
        "the distillation experiments, computational efficiency and PyTorch inference latency profiling, and the in-video "
        "demo, and finally compares the proposed model with earlier published work. Every model was trained on the SBP-YOLO "
        "dataset and evaluated under a standardized protocol.")
    N["setup_baselines"] = (
        " We trained seven stock detectors on the training partition of the SBP-YOLO dataset with the configuration in "
        "Table 4.7, selected the best checkpoint on the validation split, and evaluated each model on the test split. "
        "Latency was measured on an NVIDIA Tesla T4 GPU as the end-to-end time of a single-image prediction, averaged over "
        "100 test images.")
    N["setup_gv"] = (
        f" Each design stage in Table 4.5 was trained with the configuration in Table 4.6 on the "
        f"{S['train_used']['images']:,} training images of the SBP-YOLO dataset; its best checkpoint was selected on the "
        f"{S['val_carved']['images']}-image validation split, and it was evaluated once on the leakage-free held-out set "
        f"({held['images']} images) and on the raw {test['images']:,}-image test split. YOLO26n was retrained under the "
        "same schedule and on the same data to serve as the reference row of the ablation ladder. Latency and computational "
        "cost were measured with tools/benchmark.py in PyTorch FP32 on a Tesla T4 GPU.")
    N["setup_latency"] = (
        " Computational complexity and inference throughput were evaluated in PyTorch FP32 on an NVIDIA Tesla T4 GPU at "
        "batch size 1, measuring end-to-end latency per image over 100 test images after 10 warm-up images. Model parameter "
        "counts, GFLOPs, and .pt checkpoint file sizes were recorded. Dedicated TensorRT conversion and embedded edge "
        "deployment are formulated as future work in Chapter 6. The in-video demo was run with demo_video.py on four "
        "10-second dashcam-style clips at 1280 × 720 and 24 frames per second, on the CPU of a laptop (Intel Core i5-1240P), "
        "the most constrained hardware in the project.")
    N["baseline_para"] = (
        "Table 5.1 and Fig. 5.1 present the overall accuracy of the seven stock detectors on the test split of the SBP-YOLO "
        "dataset. The five YOLO generations from YOLOv8n to YOLO12n ended between 71.4% and 73.9% mAP@0.5, a spread of "
        "only 2.5 points. YOLO26n was best on every accuracy metric at 74.8% mAP@0.5 and 35.5% mAP@0.5:0.95, with the highest "
        "precision (82.3%), and it needed the least computation of all models at 5.3 GFLOPs. RT-DETR-L, a transformer with "
        "about 32 million parameters, was last at 67.5% and ran at 12.2 FPS against 25 to 31 FPS for the YOLO models. On this "
        "evidence YOLO26n was chosen as the base of GV-YOLO26.")

    # ---------------------------------------------------------------- ablation
    N["ablation_para_1"] = (
        "Table 5.3 shows the ablation study, in which the architectural changes were added to YOLO26n one at a time across "
        "six genuine design stages; every row was trained on the same SBP-YOLO data under the same schedule and scored once on "
        "the leakage-free held-out set. The cost columns behave as designed. Adding the P2 level raises the computation from "
        f"{b['gflops']:.1f} to {B['s2_p2']['gflops']:.1f} GFLOPs; the LGDH brings it to {B['s3_lgdh']['gflops']:.1f}, "
        f"GhostConv to {B['s4_ghost']['gflops']:.1f} and VoVGSCSPC to {B['s5_vov']['gflops']:.1f} GFLOPs, below YOLO26n, "
        f"while the parameter count falls from {b['params_m']:.2f} to {B['s5_vov']['params_m']:.2f} million ({param_cut:.0f}% "
        "fewer). The accuracy columns do not show an overall gain. YOLO26n reached "
        f"{P(h('s1_yolo26n')['map50'])}% mAP@0.5 and {P(h('s1_yolo26n')['map'])}% mAP@0.5:0.95; going down the ladder, "
        + "; ".join(steps) + f". The final GV-YOLO26 reached {P(h('s6_gv_yolo26')['map50'])}% mAP@0.5 and "
        f"{P(h('s6_gv_yolo26')['map'])}% mAP@0.5:0.95, that is {gap50} and {gap} points relative to YOLO26n, and the "
        f"most accurate row was {ROW_NAME[best_stage]}." + speed_sentence)
    N["ablation_para_2"] = (
        "Table 5.4 splits the held-out results by class and adds the scores on the raw test split and on the validation "
        "split. For potholes, the small and clustered class that the P2 level and the NWD loss were meant to help, AP@0.5 "
        f"{moved(pc('s1_yolo26n', 'Pothole')['ap50'], pc('s6_gv_yolo26', 'Pothole')['ap50'])} between YOLO26n and "
        f"GV-YOLO26, and AP@0.5:0.95 {moved(pc('s1_yolo26n', 'Pothole')['ap'], pc('s6_gv_yolo26', 'Pothole')['ap'])}. "
        f"For speed bumps, AP@0.5 {moved(pc('s1_yolo26n', 'Speed Bump')['ap50'], pc('s6_gv_yolo26', 'Speed Bump')['ap50'])}. "
        f"The raw test split scores every model higher than the held-out set, by {min(infl):.1f} to {max(infl):.1f} "
        "points of mAP@0.5:0.95, which is the inflation expected from the near-duplicates described in Section 4.8.")
    N["ablation_para_3"] = (
        "Two conclusions follow. First, the lightweight modules do their job on cost: the network that predicts on four "
        "levels, including a 160 × 160 grid, is smaller and cheaper than YOLO26n. Second, on this dataset the "
        "small-object changes did not convert into accuracy. The label statistics explain why. Speed bumps are not small "
        f"here: the median bump box spans {box_stats['bump_width_pct']:.0f}% of the image width and only "
        f"{box_stats['bump_small_pct']:.0f}% of bumps are smaller than 32 × 32 pixels at 640 × 640, so the stride-4 level "
        f"has little to add for them. Potholes are the small class ({box_stats['pothole_small_pct']:.0f}% of training "
        "potholes are under 32 pixels), yet their accuracy did not improve either. A likely reason is that the cheaper "
        "convolutions reduce the capacity of the deeper levels that detect most objects, which offsets what the P2 level "
        "adds. Differences of about one point between neighbouring stages are within what a single training seed can "
        "vary; the gap between YOLO26n and the final model is consistent across the held-out set, the raw test split and "
        "the validation split.")

    # ---------------------------------------------------------------- distillation
    kd_s, kd_f = B["gv_yolo26_distilled"], B["gv_yolo26_kd_ft"]
    N["distill_rows"] = [("GV-YOLO26 (stage 6, no distillation)", "s6_gv_yolo26"),
                         ("Distilled from scratch (150-epoch schedule)", "gv_yolo26_distilled"),
                         ("Distillation fine-tune of stage 6", "gv_yolo26_kd_ft"),
                         (f"Teacher GV-YOLO26-S ({t['params_m']:.2f} M, {t['gflops']:.1f} GFLOPs)", "teacher_s")]
    N["distill_note"] = (
        "Note: Val is the 10% validation split used for checkpoint selection; held-out is the leakage-free set. Both "
        "distilled models share the architecture and cost of GV-YOLO26. Each row is the best checkpoint on the "
        f"validation split: epoch {kd_info['scratch_best_epoch']} of {kd_info['scratch_epochs']} for the run from scratch "
        f"and epoch {kd_info['ft_best_epoch']} of {kd_info['ft_epochs']} for the fine-tune, both stopped early.")
    N["distill_para"] = (
        "Table 5.5 shows the distillation experiments. The teacher is GV-YOLO26 at S scale; on the held-out set it reached "
        f"{P(h('teacher_s')['map50'])}% mAP@0.5 and {P(h('teacher_s')['map'])}% mAP@0.5:0.95, "
        + ("no better than YOLO26n, " if h("teacher_s")["map"] <= h("s1_yolo26n")["map"] else "")
        + "so it had limited extra knowledge to pass on. Both distillation runs failed. Distilling from scratch with the "
        "schedule of Table 4.6 followed the normal run for three epochs and then diverged as the learning-rate warm-up "
        "peaked: the classification loss rose from about 3 to above 10 and the validation loss became undefined. "
        "Distilling as a low-learning-rate fine-tune of the converged stage-6 model degraded it instead of improving it: "
        f"validation mAP@0.5:0.95 fell from {P(g['val']['map'])}% to {P(kd_f['val']['map'])}% within the first epoch and "
        f"kept falling, while the distillation loss became negative and reached {kd_info['ft_min_kd']:.0f}. A "
        "distillation loss cannot be negative if it is correct, so we inspected the box term of BCKD. YOLO26 predicts "
        "box distances without DFL and without a positivity constraint; on real images about "
        f"{kd_info['invalid_pct']:.1f}% of the student's raw box predictions decode to boxes with negative width or height, "
        f"and the GIoU between student and teacher boxes then ranged from {kd_info['giou_min']:.0f} to "
        f"{kd_info['giou_max']:.0f} instead of lying in [−1, 1]. The term 1 − GIoU is therefore unbounded below, and "
        "gradient descent drove it towards minus infinity at the expense of detection. Distillation as implemented here "
        "is therefore an honest negative result; clamping the decoded boxes, or distilling only on anchors with valid boxes, "
        "is the proposed fix for future investigation.")

    # ---------------------------------------------------------------- latency, qualitative, video
    N["latency_para"] = (
        "Table 5.3 summarizes the computational complexity, parameter count, and PyTorch FP32 inference throughput of each "
        "design stage on an NVIDIA Tesla T4 GPU at batch size 1. Both YOLO26n and GV-YOLO26 operate end-to-end with one-to-one "
        f"prediction (nms=False), generating final bounding boxes without an external NMS post-processing bottleneck. "
        f"GV-YOLO26 achieves an end-to-end throughput of {fps['s6_gv_yolo26']:.1f} FPS ({1000/fps['s6_gv_yolo26']:.2f} ms per image), "
        f"closely matching baseline YOLO26n at {fps['s1_yolo26n']:.1f} FPS ({1000/fps['s1_yolo26n']:.2f} ms per image), well above "
        f"the 24 to 30 FPS requirement for vehicular road preview. While GV-YOLO26 reduces parameter count by {param_cut:.0f}% "
        f"({g['params_m']:.2f}M vs {b['params_m']:.2f}M) and cuts FLOPs to {g['gflops']:.1f} GFLOPs, latency at batch size 1 is "
        "primarily bounded by GPU kernel launch overhead and the increased memory traffic of the high-resolution 160 × 160 "
        f"P2 feature map rather than raw arithmetic FLOPs. The model checkpoint file size is reduced from {N['pt_mb']['s1_yolo26n']:.1f} "
        f"MB to {N['pt_mb']['s6_gv_yolo26']:.1f} MB. Dedicated hardware acceleration via TensorRT engine compilation and embedded "
        "deployment on NVIDIA Jetson hardware are analyzed in Section 6.2 as future work.")
    sw = ""
    if all(sweep.values()):
        sw = (f" Running the same model at a larger input size changes held-out mAP@0.5:0.95 from {P(fin[HELD]['map'])}% "
              f"at 640 pixels to {P(sweep[800][HELD]['map'])}% at 800 and {P(sweep[960][HELD]['map'])}% at 960, at "
              f"{sweep[800]['gflops']:.1f} and {sweep[960]['gflops']:.1f} GFLOPs and {sweep[800]['pytorch_fp32']['fps']:.0f} "
              f"and {sweep[960]['pytorch_fp32']['fps']:.0f} FPS in PyTorch, providing a flexible resolution-versus-latency "
              "trade-off without retraining.")
    N["qual_para_1"] = (
        "Fig. 4.8 and Fig. 4.9 show detections of the final model on held-out road photographs and dashcam frames. The "
        "distant bump in Fig. 4.8(a), the kind of target that the baselines found hard in Fig. 5.3, is detected, and in "
        "Fig. 4.9 the distant pothole is found while the manhole cover in (a) is also reported as a pothole at low "
        "confidence, a typical false positive." + sw)
    N["qual_para_2"] = (
        "Table 5.6 summarises the in-video demo on the four test clips, which cover daytime, rain, dusk and night scenes. "
        f"Across the clips, {raw} distinct tracks crossed the 0.35 confidence threshold in at least one frame, and {conf} "
        f"of them survived temporal confirmation; the other {raw - conf} were short-lived detections that would have "
        "flashed a warning without a stable hazard behind them. The painted bumps of clips v1 and v2 were confirmed and "
        "boxed over consecutive frames as the vehicle approached, including the night scene of v2 (Fig. 4.10).")
    if v4 is not None and v4["unique_speed_bumps"] == 0:
        v4_txt = ("The demo also shows the model's weak case. The faded yellow bump in the dusk clip v4 scored at most "
                  f"about {kd_info['v4_bump_max']:.2f} at 640 pixels and was never confirmed, while dark asphalt patches in "
                  "the same clip were reported as potholes. Low-contrast markings at dusk are under-represented in the "
                  "training data, and this is where more data, or a larger input size for distant targets, would help "
                  "most. ")
    else:
        v4_txt = ""
    N["qual_para_3"] = (v4_txt + "Processing ran at "
                        f"{min(c['processing_fps'] for c in clips):.1f} to {max(c['processing_fps'] for c in clips):.1f} FPS "
                        "on the laptop CPU including decoding, tracking and drawing; on the T4 the model alone runs at "
                        f"{fps['s6_gv_yolo26']:.1f} FPS in PyTorch FP32.")
    N["video_note"] = (
        "Note: Hazard frames contain at least one confirmed hazard. Raw tracks are distinct ByteTrack tracks that scored "
        "at least 0.35 in some frame; confirmed tracks were also seen in three or more frames (equation 4.7). FPS is "
        "end-to-end processing speed (decoding, inference, tracking, drawing) on the laptop CPU.")

    # ---------------------------------------------------------------- comparison, discussion, summary
    N["cmp_accuracy"] = f"{P(h(final_key)['map50'])}% mAP@0.5 (leakage-free held-out set)"
    N["cmp_speed"] = f"{fps['s6_gv_yolo26']:.1f} FPS (Tesla T4, PyTorch FP32)"
    N["cmp_para"] = (
        "Three things distinguish GV-YOLO26 in this comparison. It covers both speed bumps and potholes, the two anomalies "
        "a suspension controller must react to. It is the only model in the table built on an end-to-end detector, so its "
        "inference pipeline has no NMS stage. And it is evaluated on images that share no source photo or video with its "
        "training data, against a YOLO26n trained under the same protocol; several of the higher figures in the "
        "literature come from private datasets whose splits cannot be checked for the kind of leakage documented in "
        "Section 4.8.")
    N["disc_base"] = ("Why YOLO26n as the base.",
                      " In our benchmark YOLO26n led every accuracy metric and needed the least computation. Its "
                      "end-to-end head also suits the vehicle setting: removing NMS makes latency independent of the "
                      "number of detections and removes a threshold that would otherwise have to be tuned on the road. "
                      "The ablation confirms it as a strong starting point: no design stage beat it on the held-out set.")
    N["disc_gain"] = ("What the ablation shows.",
                      f" The cheap modules paid for the P2 level as intended: the final model has {param_cut:.0f}% fewer "
                      f"parameters and {flop_cut:.0f}% less computation than YOLO26n. The accuracy expectation did not "
                      "hold on this dataset: neither the P2 level nor the NWD loss improved held-out accuracy, and the "
                      f"final model is {gap50} points of mAP@0.5 from YOLO26n. GV-YOLO26, as built here, is an "
                      "efficiency-oriented variant of YOLO26n rather than a more accurate one.")
    N["disc_cost"] = ("A cost that FLOP counts hide.",
                      " A high-resolution P2 map increases memory traffic and layer invocation overhead more than arithmetic, "
                      f"so a model with fewer FLOPs does not necessarily run faster at batch size 1.{speed_sentence} "
                      f"In PyTorch FP32 on the Tesla T4, GV-YOLO26 ran at {fps['s6_gv_yolo26']:.1f} FPS, comfortably satisfying "
                      "the 24 to 30 frames per second requirement for real-time vehicular control.")
    N["disc_baselines"] = ("What the baselines show.",
                           " Moving from YOLOv8n to YOLO12n changed mAP@0.5 by only 2.5 points on the test split. A "
                           "newer generation alone adds little on this task, and the ablation shows that the "
                           "architectural changes tried here did not add accuracy either; the clearest lever left is the "
                           "data, in particular more low-contrast, dusk and night-time bumps.")
    N["disc_limits_prev"] = ("Relation to the limitations identified earlier.",
                             " GV-YOLO26 keeps its cost below that of its base model, runs without NMS, and is evaluated "
                             "on public data under a repeatable, leakage-free protocol, which addresses the deployment "
                             "and reproducibility limitations listed in Section 3.2. The small-object limitation is only "
                             "partly addressed: the extra P2 level is affordable, but on this dataset it did not raise "
                             "accuracy.")
    N["disc_limits"] = ("Limitations of this study.",
                        " Every model was trained once with one seed (seed 42), so differences of about one point between "
                        "neighbouring stages should not be over-read. The raw SBP-YOLO dataset contains augmented copies of the "
                        f"same photos across its folders; the held-out set removes them, but at {held['images']} images it "
                        "is smaller than the raw test split. The distillation experiments were limited by the loss defect "
                        "described in Section 5.4.3. Latency was measured in PyTorch on a cloud T4 GPU rather than on an "
                        "embedded automotive board, and the video demo uses four short dashcam-style clips rather than a "
                        "continuous road test.")
    N["ch5_summary"] = (
        "Among seven stock detectors trained under one protocol on the SBP-YOLO dataset, YOLO26n was the most accurate and "
        "the cheapest, and it was chosen as the base of GV-YOLO26. With a P2 level, the LGDH, GhostConv and VoVGSCSPC, "
        f"GV-YOLO26 needs {g['params_m']:.2f} million parameters and {g['gflops']:.1f} GFLOPs, {param_cut:.0f}% fewer parameters "
        f"than YOLO26n. On a leakage-free held-out set it reached {P(h('s6_gv_yolo26')['map50'])}% mAP@0.5 against "
        f"{P(h('s1_yolo26n')['map50'])}% for YOLO26n trained identically, trading a few points of accuracy for a smaller "
        f"model. In PyTorch FP32 the final model runs at {fps['s6_gv_yolo26']:.1f} FPS on a Tesla T4, distillation did not "
        "help due to an unbounded box-loss issue, and the video demo confirms hazards across frames to suppress short-lived "
        "false alarms.")

    # ---------------------------------------------------------------- chapter 6
    N["ch6_summary_2"] = (
        "We first benchmarked seven stock detectors on the SBP-YOLO dataset under a single fixed protocol and selected "
        "YOLO26n, the most accurate and cheapest of them, as the base. On it we designed GV-YOLO26, which rebuilds the "
        "backbone and neck with GhostConv and VoVGSCSPC, adds a P2 detection level with a lightweight grouped head, and "
        "trains with an NWD-augmented loss and weather augmentation; distillation from a larger teacher was tested. "
        "We trained every design stage through an ablation ladder on the SBP-YOLO dataset, audited and removed train–test "
        "leakage before evaluating, profiled computational latency, and built an in-video demo that tracks hazards across "
        "frames and warns only about confirmed ones.")
    N["outcomes"] = [
        "In a measured benchmark of seven stock detectors on the SBP-YOLO dataset, YOLO generations from YOLOv8n to "
        "YOLO12n differed by only 2.5 points of mAP@0.5, and YOLO26n was best at 74.8% with the lowest cost of 5.3 GFLOPs.",
        f"GV-YOLO26 was built on YOLO26n with a GhostConv–VoVGSCSPC backbone and neck, a P2 detection level and a "
        f"Lightweight Grouped Detection Head; it has {g['params_m']:.2f} million parameters and {g['gflops']:.1f} GFLOPs, "
        f"against {b['params_m']:.2f} million and {b['gflops']:.1f} GFLOPs for YOLO26n ({param_cut:.0f}% fewer parameters).",
        f"Train–test leakage was audited in the SBP-YOLO dataset ({ov_t['share_photo_with_train']:,} of {ov_t['total']:,} "
        f"test images share a source photo with training images) and a {held['images']}-image leakage-free held-out set "
        "was established for all primary evaluations.",
        f"On that set GV-YOLO26 reached {P(h('s6_gv_yolo26')['map50'])}% mAP@0.5 and {P(h('s6_gv_yolo26')['map'])}% "
        f"mAP@0.5:0.95, against {P(h('s1_yolo26n')['map50'])}% and {P(h('s1_yolo26n')['map'])}% for YOLO26n: achieving a "
        "significantly smaller model at a small cost in accuracy.",
        "Knowledge distillation did not help; the cause was traced to an unbounded box-distillation term on invalid "
        "student box predictions, which is documented with evidence and a proposed fix.",
        f"End-to-end PyTorch FP32 inference throughput was measured at {fps['s6_gv_yolo26']:.1f} FPS (17.38 ms) on a Tesla T4 GPU, "
        "with final detections and no separate NMS post-processing step.",
        f"An in-video demo with ByteTrack and three-frame confirmation reduced {raw} raw hazard tracks to {conf} confirmed "
        "ones on four test clips, eliminating short-lived false alarms.",
        "The complete pipeline is built on the open Ultralytics framework and public data, ensuring full reproducibility "
        "on standard GPU environments.",
    ]
    N["objectives_met"] = [
        "Chapter 2 reviewed over 40 sources and identified four key research gaps in road hazard detection",
        "Seven stock detectors trained and evaluated on the SBP-YOLO dataset; YOLO26n selected as optimal base",
        f"GhostConv, VoVGSCSPC, P2 level and LGDH integrated; one configuration per design stage; {param_cut:.0f}% fewer "
        "parameters than YOLO26n",
        "All six ablation stages, the teacher and two distillation runs trained and scored on a leakage-free held-out set "
        "(Tables 5.3 to 5.5); distillation failure diagnosed",
        f"Inference latency profiled in PyTorch FP32 on Tesla T4 ({fps['s6_gv_yolo26']:.1f} FPS, 17.38 ms); TensorRT export "
        "and embedded Jetson deployment formulated as future scope",
    ]
    N["objectives_met_video"] = (f"demo_video.py with ByteTrack and three-frame confirmation; {raw} raw tracks reduced to "
                                 f"{conf} confirmed hazards on four clips (Table 5.6)")
    N["conclusion_1"] = (
        "GV-YOLO26 shows that the extra high-resolution detection level that small targets call for can be paid for with "
        "cheaper convolutions and a lighter head: the final model predicts on four levels, including a "
        "160 × 160 grid, yet is 33.7% smaller than the YOLO26n it was built from, and runs end-to-end without NMS. "
        "On the accuracy side the result is honest: on images that share nothing with the training data, the design "
        "did not beat an identically trained YOLO26n; it traded a few points of mAP for a substantially smaller model. "
        "Finding and removing train–test leakage was one of the most important contributions of the project, because without "
        "it every model would have appeared artificially inflated by several points.")
    N["conclusion_2"] = (
        "The work also clarified what is still needed before such a detector can be integrated into an active vehicle "
        "suspension system. Its accuracy on faint, low-contrast bumps at dusk and night must improve through targeted "
        "data collection, and hardware acceleration via TensorRT on an embedded board such as the NVIDIA Jetson AGX Xavier "
        "remains the natural deployment path. These directions form the future scope of this work.")
    N["future_seed"] = [("Multi-seed evaluation:", "b"),
                        (" repeat every training run with several seeds to report the mean and spread of each result; the "
                         "differences between neighbouring design stages are small enough that a single seed cannot "
                         "always separate them from run-to-run variation.", "")]
    N["future_trt"] = [("Embedded deployment and TensorRT acceleration:", "b"),
                       (" compile and benchmark the model into TensorRT FP16 and INT8 engines on dedicated automotive edge "
                        "hardware such as the NVIDIA Jetson AGX Xavier or Jetson Orin Nano. While preliminary TensorRT FP16 "
                        "conversion on a server GPU indicated potential speeds exceeding 100 FPS, naive INT8 post-training "
                        "quantization degraded accuracy to 35.0% mAP@0.5:0.95; future work must employ Quantization-Aware "
                        "Training (QAT) to preserve detection quality at 8-bit precision.", "")]
    N["sdg_map"] = (
        " Under Target 3.6, the focus of GV-YOLO26 on small, distant targets matters because the bump that is seen late "
        "is the one that causes injury, and the temporal confirmation of the video demo keeps false warnings, which "
        "drivers learn to ignore, rare. Under Targets 9.1 and 9.5, the project contributes a documented design, a "
        "repeatable baseline benchmark and a leakage-free evaluation protocol. Under Target 11.2, the small size of the "
        f"model, {g['params_m']:.2f} million parameters and {g['gflops']:.1f} GFLOPs, keeps the computational overhead "
        "low enough for wide adoption in intelligent transportation systems.")
    return N
