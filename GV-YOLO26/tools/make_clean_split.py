"""Build a leakage-free held-out set and dataset statistics (Section 4.8, Table 4.4).

The extended dataset contains augmented copies of the same photo (e.g. 'add_gasuss_noise_1386_jpg.rf.<hash>.jpg' in
train/ and '1386_jpg.rf.<hash>.jpg' in test/) and consecutive frames of the same video in different folders. An image
of test/ or valid/ is kept in the clean held-out set only if neither its source photo nor its source video/sequence
appears anywhere in train/. The clean set is written as a list of image paths and a data YAML whose 'test' entry
points to it.

Example:
    python tools/make_clean_split.py --root /kaggle/input/.../speed12 --data data/data.yaml --out data
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import yaml

IMG = {".jpg", ".jpeg", ".png"}
SEQ = [r"^(MVI_\d+)", r"^(G\d{3})\d+$", r"^(Vid_\d+)_", r"^(Vid_[A-Za-z]+_[a-z]+)_", r"^(VIDEO\d+)_",
       r"^(DASH_CAM.{0,60}?)-?_mp\d", r"^(image_\d+)_\d+$", r"^(V\d+F_.{0,60}?Da)",
       r"^((?:AM|PM)_[A-Za-z]+_CI\d+_\d{8})", r"^(20\d{6})\d{6}"]  # dashcam drive: camera + date


AUG_PREFIXES = ("add_motion_blur_", "add_gasuss_noise_", "add_snow_weather_", "add_rain_weather_", "random_bright_",
                "random_contrast_", "random_saturatio_", "random_flip_vert_", "random_flip_hori_", "center_crop_",
                "resizeDown_keep__")


def photo_key(name):
    """Source photo: drop the Roboflow '.rf.<hash>' suffix, the '_jpg' marker and augmentation-tag prefixes."""
    b = name.split(".rf.")[0] if ".rf." in name else name.rsplit(".", 1)[0]
    b = re.sub(r"_(jpe?g|png|JPE?G|PNG)$", "", b)
    stripped = True
    while stripped:  # tags can be stacked, e.g. add_motion_blur_po_1845
        stripped = False
        for t in AUG_PREFIXES:
            if b.startswith(t):
                b, stripped = b[len(t):], True
    return b.replace("_flipped", "")


def seq_key(name):
    """Source video or image sequence (frames of one clip / one GoPro group share a key)."""
    p = photo_key(name)
    for rx in SEQ:
        m = re.match(rx, p)
        if m:
            return m.group(1)
    return p


def images(d):
    return sorted(p for p in Path(d).glob("*") if p.suffix.lower() in IMG)


def label_stats(paths, names):
    inst, imgs_with = Counter(), Counter()
    background = 0
    for p in paths:
        lp = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        cls = [int(line.split()[0]) for line in lp.read_text().splitlines() if line.strip()] if lp.exists() else []
        if not cls:
            background += 1
        inst.update(cls)
        imgs_with.update(set(cls))
    return dict(images=len(paths), background=background,
                instances={names[c]: inst[c] for c in sorted(names)},
                images_with={names[c]: imgs_with[c] for c in sorted(names)})


def main(opt):
    root = Path(opt.root).resolve()
    data = yaml.safe_load(open(opt.data))
    names = {int(k): v for k, v in data["names"].items()}
    train_all = images(root / "train/images")
    tr_photo = {photo_key(p.name) for p in train_all}
    tr_seq = {seq_key(p.name) for p in train_all}

    clean, report = [], {}
    for split in ["test", "valid"]:
        files = images(root / split / "images")
        photo_ok = [p for p in files if photo_key(p.name) not in tr_photo]
        seq_ok = [p for p in photo_ok if seq_key(p.name) not in tr_seq]
        clean += seq_ok
        report[f"{split}_overlap"] = dict(total=len(files), share_photo_with_train=len(files) - len(photo_ok),
                                          share_sequence_with_train=len(files) - len(seq_ok), clean=len(seq_ok))

    out = Path(opt.out)
    (out / "clean_holdout.txt").write_text("\n".join(str(p) for p in clean) + "\n")
    clean_yaml = dict(data, test=str((out / "clean_holdout.txt").resolve()))
    yaml.safe_dump(clean_yaml, open(out / "data_clean.yaml", "w"), sort_keys=False)
    valid_yaml = dict(data, test=str(root / "valid/images"))
    yaml.safe_dump(valid_yaml, open(out / "data_valid.yaml", "w"), sort_keys=False)

    def listed(txt):
        return [Path(x) for x in Path(txt).read_text().split() if x]

    folders = {f: len(images(root / f / "images")) for f in ("train", "valid", "test")}
    report["folders"], report["total_images"] = folders, sum(folders.values())
    report["splits"] = {
        "train_used": label_stats(listed(data["train"]), names),
        "val_carved": label_stats(listed(data["val"]), names),
        "test": label_stats(images(root / "test/images"), names),
        "valid_folder": label_stats(images(root / "valid/images"), names),
        "clean_holdout": label_stats(clean, names),
    }
    Path(opt.stats).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="dataset root with train/, valid/ and test/")
    p.add_argument("--data", default="data/data.yaml")
    p.add_argument("--out", default="data")
    p.add_argument("--stats", default="results/dataset_stats.json")
    main(p.parse_args())
