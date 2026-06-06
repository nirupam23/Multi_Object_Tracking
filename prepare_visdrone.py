#!/usr/bin/env python3
"""
Convert the VisDrone-MOT dataset into YOLO detection format.

VisDrone-MOT layout (per split):
    <root>/sequences/<seq_name>/0000001.jpg ...
    <root>/annotations/<seq_name>.txt

Annotation columns (one detection per line):
    frame_index, target_id, bbox_left, bbox_top, bbox_w, bbox_h,
    score, object_category, truncation, occlusion

VisDrone object_category ids:
    0 ignored   1 pedestrian   2 people     3 bicycle   4 car      5 van
    6 truck     7 tricycle     8 awning-tri 9 bus      10 motor   11 others

We are targeting VEHICLES, so all wheeled-vehicle categories
{3 bicycle, 4 car, 5 van, 6 truck, 9 bus, 10 motor, 7 tricycle, 8 awning-tricycle}
are merged into a single YOLO class 0 = "vehicle". (Edit DEFAULT_MAP below to change
this — e.g. {1:0, 2:0} for the 'person' variant of the task, or remove 3:0 to
exclude human-powered bicycles.)

Output (YOLO standard):
    <out>/images/{train,val}/<seq>__<frame>.jpg   (symlink by default)
    <out>/labels/{train,val}/<seq>__<frame>.txt   (normalised xywh)
    <out>/visdrone_vehicle.yaml

Usage:
    python scripts/prepare_visdrone.py \
        --train-root /data/VisDrone2019-MOT-train \
        --val-root   /data/VisDrone2019-MOT-val \
        --out        /data/visdrone_yolo
"""

import argparse
import os
import glob
import shutil

import cv2

# VisDrone category id -> output YOLO class id. Everything else is dropped.
# --- VEHICLE target (active) -----------------------------------------------
# All wheeled vehicles incl. two-wheelers -> class 0 "vehicle":
# bicycle, car, van, truck, bus, motor, tricycle, awning-tricycle
DEFAULT_MAP = {3: 0, 4: 0, 5: 0, 6: 0, 9: 0, 10: 0, 7: 0, 8: 0}
# remove `3: 0` to exclude bicycles (human-powered) if you want motorised-only.
#
# --- PERSON target (alternative) -------------------------------------------
# To target persons instead, comment out the vehicle map above and use:
# DEFAULT_MAP = {1: 0, 2: 0}   # pedestrian, people -> person
# (and run with --names person)


def convert_split(root, split, out_dir, class_map, copy=False):
    seq_dir = os.path.join(root, "sequences")
    ann_dir = os.path.join(root, "annotations")
    img_out = os.path.join(out_dir, "images", split)
    lbl_out = os.path.join(out_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    sequences = sorted(
        d for d in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, d))
    )
    n_imgs = n_boxes = 0

    for seq in sequences:
        ann_path = os.path.join(ann_dir, seq + ".txt")
        if not os.path.isfile(ann_path):
            continue

        # group annotation rows by frame index
        per_frame = {}
        with open(ann_path) as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                fidx = int(parts[0])
                left, top, w, h = map(float, parts[2:6])
                cat = int(parts[7])
                if cat not in class_map or w <= 0 or h <= 0:
                    continue
                per_frame.setdefault(fidx, []).append(
                    (class_map[cat], left, top, w, h)
                )

        frames = sorted(glob.glob(os.path.join(seq_dir, seq, "*.jpg")))
        for fpath in frames:
            fname = os.path.basename(fpath)
            raw = os.path.splitext(fname)[0]
            try:
                fidx = int(raw)
            except ValueError:
                # skip junk / Windows duplicate files like "0000388 (1).jpg"
                print(f"  skipping non-frame file: {fname}")
                continue
            stem = f"{seq}__{fidx:07d}"

            img = cv2.imread(fpath)
            if img is None:
                continue
            H, W = img.shape[:2]

            # write label file (may be empty -> negative/background frame)
            label_lines = []
            for cls, left, top, bw, bh in per_frame.get(fidx, []):
                xc = (left + bw / 2) / W
                yc = (top + bh / 2) / H
                nw, nh = bw / W, bh / H
                # clip to [0,1]
                xc, yc = min(max(xc, 0), 1), min(max(yc, 0), 1)
                nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
                label_lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                n_boxes += 1

            with open(os.path.join(lbl_out, stem + ".txt"), "w") as out:
                out.write("\n".join(label_lines))

            dst_img = os.path.join(img_out, stem + ".jpg")
            if copy:
                shutil.copy(fpath, dst_img)
            else:
                if os.path.lexists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(fpath), dst_img)
            n_imgs += 1

    print(f"[{split}] {len(sequences)} sequences, {n_imgs} images, {n_boxes} boxes")
    return n_imgs


def write_yaml(out_dir, names, have_train=True):
    yaml_path = os.path.join(out_dir, f"visdrone_{names[0]}.yaml")
    # If no train split was prepared (e.g. only the val set is downloaded yet),
    # point `train` at the val images so the yaml is still usable for a quick
    # smoke-test fine-tune. Replace with the real train split once available.
    train_path = "images/train" if have_train else "images/val"
    with open(yaml_path, "w") as fh:
        fh.write(f"path: {os.path.abspath(out_dir)}\n")
        fh.write(f"train: {train_path}\n")
        fh.write("val: images/val\n")
        fh.write(f"nc: {len(names)}\n")
        fh.write("names:\n")
        for i, n in enumerate(names):
            fh.write(f"  {i}: {n}\n")
    print(f"wrote {yaml_path}")
    if not have_train:
        print("  NOTE: no --train-root given, so 'train:' points at the val "
              "images. Re-run with --train-root once the train set is "
              "downloaded for a real fine-tune.")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default=None,
                    help="VisDrone2019-MOT-train root (optional; omit if you "
                         "only have the val set so far)")
    ap.add_argument("--val-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (use on Windows)")
    ap.add_argument("--names", nargs="+", default=["vehicle"])
    args = ap.parse_args()

    have_train = args.train_root is not None
    if have_train:
        convert_split(args.train_root, "train", args.out, DEFAULT_MAP, args.copy)
    convert_split(args.val_root, "val", args.out, DEFAULT_MAP, args.copy)
    write_yaml(args.out, args.names, have_train=have_train)


if __name__ == "__main__":
    main()
